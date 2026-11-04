"""
模型定义: 条件生成器、判别器与 EMA

- Generator: 渐进式 CondUpBlock 上采样 + FiLM 条件注入 + 可学习噪声注入
              4×6 → 8×12 → 16×24 → 35×90, 各级分辨率均接收文本条件
- Discriminator: 图像 → 真/假 + 4 字符分类 (AC-GAN),
              逐位辅助分类头强制空间分离字符, 缓解粘连
- EMA: 生成器权重的指数移动平均, 用于稳定采样与推理

噪声注入说明:
  噪声增益为逐通道可学习参数, 训练与推理时均注入随机噪声,
  配合同标签双噪声对比损失, 保证同标签不同采样天然多样。
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from config import *


# ═══════════════════════════════════════════════════════════
#  噪声注入 (StyleGAN 式逐通道可学习增益)
# ═══════════════════════════════════════════════════════════

class NoiseInjection(nn.Module):
    """
    逐通道可学习噪声注入 (StyleGAN 式)。
    训练和推理均注入随机噪声, 增益可学习 — 让同标签不同噪声采样天然多样。

    与固定噪声标准差的区别:
      - 每通道独立增益 (初始为 NOISE_STD, 保证初期即有随机性)
      - 训练/推理一致生效, 避免推理时退化为确定性映射
    """

    def __init__(self, channels, init_std=None):
        super().__init__()
        if init_std is None:
            init_std = NOISE_STD
        self.gain = nn.Parameter(torch.full((channels,), float(init_std)))

    def forward(self, x):
        noise = torch.empty(x.shape, device=x.device, dtype=x.dtype).normal_(0, 1)
        return x + noise * self.gain.view(1, -1, 1, 1)


# ═══════════════════════════════════════════════════════════
#  Self-Attention
# ═══════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    """轻量 Self-Attention: 帮助生成器捕捉字符之间的空间关系"""

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


# ═══════════════════════════════════════════════════════════
#  文本编码
# ═══════════════════════════════════════════════════════════

class TextEmbedding(nn.Module):
    """将 4 字符的标签编码为条件向量 (用于 Generator)"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.fc = nn.Sequential(
            nn.Linear(EMBED_DIM * CAPTCHA_LENGTH, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, labels):
        B = labels.shape[0]
        e = self.embed(labels)
        e = e.view(B, -1)
        return self.fc(e)


# ═══════════════════════════════════════════════════════════
#  Generator: 渐进式生成 + FiLM 条件注入 + 噪声
# ═══════════════════════════════════════════════════════════

class CondUpBlock(nn.Module):
    """
    带 FiLM 条件注入的上采样块。
    每个块: Upsample → Conv → BN → FiLM → ReLU → Conv → BN → ReLU → Noise,
    文本条件通过 scale+bias 注入每一级分辨率。
    """

    def __init__(self, in_ch, out_ch, cond_dim, scale_h=2, scale_w=2,
                 noise_std=None):
        super().__init__()
        if noise_std is None:
            noise_std = NOISE_STD

        if scale_h == 1 and scale_w == 1:
            self.upsample = nn.Identity()
        else:
            self.upsample = nn.Upsample(scale_factor=(scale_h, scale_w),
                                        mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # FiLM: 条件投影为 per-channel scale + bias
        self.cond_scale = nn.Linear(cond_dim, out_ch)
        self.cond_bias = nn.Linear(cond_dim, out_ch)

        # 逐通道可学习噪声增益, 训练/推理均注入
        self.noise_gain = nn.Parameter(torch.full((out_ch,), float(noise_std)))

    def forward(self, x, cond):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)

        # FiLM 调制: scale=1+α, bias=β
        scale = self.cond_scale(cond).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        bias = self.cond_bias(cond).unsqueeze(-1).unsqueeze(-1)
        x = x * (1.0 + scale) + bias
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        noise = torch.empty(x.shape, device=x.device, dtype=x.dtype).normal_(0, 1)
        x = x + noise * self.noise_gain.view(1, -1, 1, 1)
        return x


class Generator(nn.Module):
    """
    渐进式条件生成器:
    - 输入: 噪声 z (B, 256) + 文本条件 (B, 256)
    - 输出: 90×35 灰度图 (B, 1, 35, 90)

    结构:
    - 小初始化 Linear(512→6144) → 4×6 空间特征, 避免单层大投影记忆训练集
    - 4×6 最低分辨率处 Self-Attention 捕捉全局结构
    - CondUpBlock ×3 逐级上采样并注入 FiLM 条件
    - 无 Skip Connections, 防止编码器过拟合特征被解码器直接复用
    """

    def __init__(self):
        super().__init__()
        self.text_embed = TextEmbedding()
        C = 64  # base channels

        # ── 噪声+文本 → 共享条件 ──
        self.cond_proj = nn.Sequential(
            nn.Linear(LATENT_DIM + 256, 512),
            nn.ReLU(inplace=True),
        )

        # ── 小初始化: 512 → 256 × 4 × 6 = 6144 ──
        init_ch = C * 4  # 256
        self.init_h, self.init_w = 4, 6
        self.init_fc = nn.Linear(512, init_ch * self.init_h * self.init_w)

        # ── Bottleneck 处理 (4×6) ──
        self.bottleneck = nn.Sequential(
            nn.Conv2d(init_ch, init_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_ch), nn.ReLU(inplace=True),
            nn.Conv2d(init_ch, init_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_ch), nn.ReLU(inplace=True),
        )
        self.attn = SelfAttention(init_ch)  # 4×6 → 24 tokens, 高效 attention

        # ── 渐进式上采样 (FiLM 注入在每级) ──
        # 4×6 → 8×12 → 16×24 → 35×90
        self.up1 = CondUpBlock(init_ch, C * 2, cond_dim=512,
                               scale_h=2, scale_w=2)      # 4×6 → 8×12
        self.up2 = CondUpBlock(C * 2, C, cond_dim=512,
                               scale_h=2, scale_w=2)      # 8×12 → 16×24
        # 末级: 16×24 → 35×90 (非均匀上采样), 高分辨率细节同样注入噪声
        self.up3 = nn.Sequential(
            nn.Upsample(size=(35, 90), mode='bilinear', align_corners=False),
            nn.Conv2d(C, C // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 2), nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 4), nn.ReLU(inplace=True),
            NoiseInjection(C // 4),
        )

        # ── 输出层 ──
        self.final = nn.Conv2d(C // 4, 1, 3, padding=1, bias=False)

    def forward(self, z, labels):
        B = z.size(0)
        init_ch = 256  # C * 4

        # 1. 条件编码
        cond = self.text_embed(labels)                      # (B, 256)
        cond = self.cond_proj(torch.cat([z, cond], dim=1))  # (B, 512)

        # 2. 小初始化 → 最小空间特征
        x = self.init_fc(cond)                              # (B, 6144)
        x = x.view(B, init_ch, self.init_h, self.init_w)    # (B, 256, 4, 6)

        # 3. Bottleneck + Self-Attention
        x = self.bottleneck(x)
        x = self.attn(x)

        # 4. 渐进式上采样 + FiLM 条件注入
        x = self.up1(x, cond)           # (B, 128, 8, 12)
        x = self.up2(x, cond)           # (B, 64, 16, 24)
        x = self.up3(x)                 # (B, 16, 35, 90)

        # 5. 输出 → Tanh
        x = self.final(x)               # (B, 1, 35, 90)
        return torch.tanh(x).contiguous()


# ═══════════════════════════════════════════════════════════
#  Discriminator (AC-GAN + SpectralNorm + 逐位辅助分类)
# ═══════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """
    条件判别器:
    - 4 层谱归一化卷积 + InstanceNorm + LeakyReLU
    - conv4 后注入 Minibatch StdDev 通道, 帮助检测模式坍塌
    - 逐位辅助分类头: 将池化特征按宽度拆为 4 列, 每列独立分类一个字符位置,
      强制判别器从不同空间区域识别不同位置, 推动生成器分离字符 (缓解粘连)
    - ReACGAN 超球面投影: 辅助头输入 L2 归一化, 稳定 AC-GAN 训练
    """

    def __init__(self):
        super().__init__()

        # conv1: → (DIS_FEATURES, H/2, W/2)
        self.conv1 = spectral_norm(nn.Conv2d(1, DIS_FEATURES, kernel_size=4,
                                             stride=2, padding=1, bias=False))
        self.in1 = nn.InstanceNorm2d(DIS_FEATURES, affine=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # conv2: → (DIS_FEATURES*2, H/4, W/4)
        self.conv2 = spectral_norm(nn.Conv2d(DIS_FEATURES, DIS_FEATURES * 2,
                                             kernel_size=4, stride=2, padding=1, bias=False))
        self.in2 = nn.InstanceNorm2d(DIS_FEATURES * 2, affine=False)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # conv3: → (DIS_FEATURES*4, H/8, W/8)
        self.conv3 = spectral_norm(nn.Conv2d(DIS_FEATURES * 2, DIS_FEATURES * 4,
                                             kernel_size=4, stride=2, padding=1, bias=False))
        self.in3 = nn.InstanceNorm2d(DIS_FEATURES * 4, affine=False)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        # conv4: → (DIS_FEATURES*8, H/16, W/16)
        self.conv4 = spectral_norm(nn.Conv2d(DIS_FEATURES * 4, DIS_FEATURES * 8,
                                             kernel_size=4, stride=2, padding=1, bias=False))
        self.in4 = nn.InstanceNorm2d(DIS_FEATURES * 8, affine=False)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)

        # Minibatch StdDev 在 conv4 后、pool 前注入, 通道数 +1
        # AvgPool2d((1,2), stride=1) 替代 AdaptiveAvgPool2d, 提升 NPU 兼容性
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), stride=1)
        self._num_spatial_ch = DIS_FEATURES * 8 + 1  # 空间通道数
        self._spatial_h = 2
        self._spatial_w = 4  # 4 个宽度位置 ← 对应 4 个字符位置
        pooled_dim = self._num_spatial_ch * self._spatial_h * self._spatial_w

        # 判别头 (真/假)
        self.disc_head = nn.Sequential(
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(pooled_dim, 1, bias=False)),
        )

        # 逐位辅助分类头 (防字符粘连)
        pos_dim = self._num_spatial_ch * self._spatial_h
        self.aux_pos_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pos_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, NUM_CLASSES),
            )
            for _ in range(CAPTCHA_LENGTH)
        ])

    def _backbone(self, x, collect_features=False):
        """统一 backbone: 可选收集中间层特征 (供特征匹配损失使用)"""
        feats = [] if collect_features else None

        x = self.conv1(x); x = self.in1(x); x = self.act1(x)
        if collect_features: feats.append(x)

        x = self.conv2(x); x = self.in2(x); x = self.act2(x)
        if collect_features: feats.append(x)

        x = self.conv3(x); x = self.in3(x); x = self.act3(x)
        if collect_features: feats.append(x)

        x = self.conv4(x); x = self.in4(x); x = self.act4(x)
        if collect_features: feats.append(x)

        # --- Minibatch StdDev (Progressive GANs) ---
        B, C, H, W = x.shape
        mb_std = x.std(dim=0, keepdim=True).mean(dim=(1), keepdim=True)  # (1, 1, H, W)
        mb_std = mb_std.expand(B, 1, H, W)
        x = torch.cat([x, mb_std], dim=1)

        h = self.pool(x)
        # contiguous() 确保 NPU 内部格式 → 标准 NCHW 后再 reshape
        h = h.contiguous().reshape(h.size(0), -1)

        return (h, feats) if collect_features else h

    def forward(self, x, labels=None, return_features=False):
        if return_features:
            h, features = self._backbone(x, collect_features=True)
        else:
            h = self._backbone(x, collect_features=False)
            features = []

        validity = self.disc_head(h)

        # 将特征图空间拆分为 4 个位置, 每个位置独立分类
        h_spatial = h.reshape(h.size(0), self._num_spatial_ch,
                              self._spatial_h, self._spatial_w)

        aux_outputs = []
        for pos in range(CAPTCHA_LENGTH):
            pos_feat = h_spatial[:, :, :, pos:pos+1]  # (B, C+1, 2, 1)
            pos_feat = pos_feat.reshape(pos_feat.size(0), -1)  # (B, (C+1)*2)
            if NORMALIZE_AUX:
                pos_feat = F.normalize(pos_feat, p=2, dim=1)
            aux_p = self.aux_pos_heads[pos](pos_feat)  # (B, NUM_CLASSES)
            aux_outputs.append(aux_p)
        aux = torch.stack(aux_outputs, dim=1)  # (B, 4, NUM_CLASSES)

        return validity, aux, features


# ═══════════════════════════════════════════════════════════
#  EMA (指数移动平均)
# ═══════════════════════════════════════════════════════════

class EMAGenerator:
    """
    Generator 的 EMA 副本, 用于采样与保存。

    update() 同时同步可训练参数与 BN 统计量 (buffers):
    parameters() 不含 BN 的 running_mean/running_var,
    若只平滑参数会导致推理时 BN 归一化使用初始化统计量, 样本不可信。
    """

    def __init__(self, generator, decay=EMA_DECAY):
        self.ema_model = copy.deepcopy(generator).eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad = False
        for b in self.ema_model.buffers():
            b.requires_grad = False

    def update(self, generator):
        """EMA 更新: 平滑可训练参数, 直接复制 BN 统计量"""
        with torch.no_grad():
            for ema_p, g_p in zip(self.ema_model.parameters(), generator.parameters()):
                ema_p.data.mul_(self.decay).add_(g_p.data, alpha=1 - self.decay)
            for ema_b, g_b in zip(self.ema_model.buffers(), generator.buffers()):
                ema_b.data.copy_(g_b.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)

    def to(self, device):
        self.ema_model = self.ema_model.to(device)
        return self

    def eval(self):
        self.ema_model.eval()


# ═══════════════════════════════════════════════════════════
#  权重初始化
# ═══════════════════════════════════════════════════════════

def weights_init(m):
    """高斯初始化 (判别器)"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("ConvTranspose") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)


def weights_init_ortho(m):
    """正交初始化 (适用于 SpectralNorm + Hinge 的生成器)"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("ConvTranspose") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.orthogonal_(m.weight, gain=1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.orthogonal_(m.weight, gain=1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ═══════════════════════════════════════════════════════════
#  自检
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from device_utils import DEVICE
    device = DEVICE

    G = Generator().to(device).apply(weights_init)
    D = Discriminator().to(device).apply(weights_init)

    z = torch.randn(4, LATENT_DIM).to(device)
    labels = torch.randint(0, NUM_CLASSES, (4, CAPTCHA_LENGTH)).to(device)

    fake = G(z, labels)
    print(f"Generator output: {fake.shape}  [{fake.min():.2f}, {fake.max():.2f}]")

    validity, aux, features = D(fake, labels, return_features=True)
    print(f"Discriminator validity: {validity.shape}")
    print(f"Discriminator aux: {aux.shape}")
    print(f"Feature layers: {len(features)}")
    for i, f in enumerate(features):
        print(f"  layer{i+1}: {f.shape}")

    ema = EMAGenerator(G, decay=0.999)
    ema.update(G)
    print(f"\nEMA Generator created")

    param_g = sum(p.numel() for p in G.parameters() if p.requires_grad)
    param_d = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"\nParams — Generator: {param_g/1e6:.2f}M, Discriminator: {param_d/1e6:.2f}M")
