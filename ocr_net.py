"""强 OCR 锚点: 加载用户提供的 CaptchaResNet, 冻结用于字形监督.
- 36 类 (数字0-9 + 字母 a-z, 不分大小写), 索引与项目大写36类一一对应
- 输入: (B,1,35,90) tanh[-1,1] 灰度, 与项目 G 输出直接兼容
- 支持: 分类 logits + 中间特征提取 (feature matching)
"""
import torch
import torch.nn as nn

from config import NUM_CLASSES, CAPTCHA_LENGTH, IMG_HEIGHT, IMG_WIDTH

OCR_WEIGHTS = "OCR/best_captcha_resnet.pth"


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class CaptchaResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, captcha_len=CAPTCHA_LENGTH):
        super().__init__()
        self.captcha_len = captcha_len
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, captcha_len))
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        """x: (B,1,H,W) tanh[-1,1]. 返回 logits [B,4,36] 或 (logits, features).
        features: layer3 激活 [B,256,H/4,W/4] 用于感知匹配."""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        feats = self.layer3(x)            # (B,256,8,22)
        x = self.pool(feats)              # (B,256,1,4)
        x = x.squeeze(2).permute(0, 2, 1)  # (B,4,256)
        x = self.dropout(x)
        out = self.classifier(x)           # (B,4,36)
        if return_features:
            return out, feats
        return out


_ocr_model = None
_ocr_loaded = False


def load_ocr(device=None, freeze=True):
    """加载并冻结强 OCR. 返回推理模式模型 (可梯度回传, 但参数不更新)."""
    global _ocr_model, _ocr_loaded
    if device is None:
        from device_utils import DEVICE
        device = DEVICE
    if _ocr_loaded:
        return _ocr_model
    model = CaptchaResNet().to(device)
    state = torch.load(OCR_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    _ocr_model = model
    _ocr_loaded = True
    return model


def ocr_loss(logits, labels):
    """分类 CE 监督. logits: (B,4,36), labels: (B,4)."""
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))


def ocr_feature_loss(feat_fake, feat_real):
    """特征匹配: 让生成字形在 OCR 特征空间贴近真实字形.
    feat: (B,256,H,W). 匹配逐通道均值."""
    return torch.nn.functional.l1_loss(
        feat_fake.mean(dim=(2, 3)), feat_real.mean(dim=(2, 3)).detach())


def ocr_accuracy(model, images, labels):
    """整串准确率 (0-dim tensor). 4个字符全部正确才算对."""
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(-1)  # (B, 4)
        return (preds == labels).all(dim=1).float().mean()


if __name__ == "__main__":
    import glob
    from pathlib import Path
    import random
    import numpy as np
    from PIL import Image
    from config import CHARS, LATENT_DIM

    m = load_ocr()
    print("OCR loaded.")

    # 1. 真实数据准确率
    files = sorted(glob.glob('训练数据/batch_*/captcha_*.jpg'))
    random.seed(0)
    sample = random.sample(files, 3000)
    ok = 0
    tot = 0
    from device_utils import DEVICE
    for f in sample:
        a = np.array(Image.open(f).convert('L'), dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(DEVICE)
        stem = Path(f).stem.replace('captcha_', '', 1)
        clean = ''.join(c for c in stem if c.isalnum())[:4]
        lbl = torch.tensor([CHARS.index(c.upper()) for c in clean],
                           device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            pred = m(x).argmax(-1)
        ok += (pred == lbl).sum().item()
        tot += 4
    print(f'OCR on REAL data: {ok / tot * 100:.1f}% char acc')

    # 2. 当前生成图准确率
    from models import Generator
    from utils import load_model
    from config import OUTPUT_DIR
    ckpts = sorted(Path(OUTPUT_DIR).glob("checkpoint_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"未找到 checkpoint: {OUTPUT_DIR}/checkpoint_epoch_*.pt")
    G = Generator().to(DEVICE).eval()
    load_model(ckpts[-1], G, None, device=DEVICE)
    torch.manual_seed(7)
    gok = 0
    gtot = 0
    with torch.no_grad():
        for _ in range(30):
            lbl = torch.randint(0, NUM_CLASSES, (128, CAPTCHA_LENGTH), device=DEVICE)
            z = torch.randn(128, LATENT_DIM, device=DEVICE)
            fake = G(z, lbl)
            pred = m(fake).argmax(-1)
            gok += (pred == lbl).sum().item()
            gtot += 128 * 4
    print(f'OCR on CURRENT generated images: {gok / gtot * 100:.1f}% char acc')