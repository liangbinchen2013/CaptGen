"""
模型定义 (v10 架构重写版 + v12 多样性修复):
- Generator: 渐进式 CondUpBlock ×3, FiLM条件注入, 无UNet
  v10: 废除 init_proj 单层1250万参数, 改为小初始化+逐级生成
  4×6 → 8×12 → 16×24 → 35×90, 每级注入文本条件
- Discriminator: 图像 → 真/假 + 4字符分类 (AC-GAN)
  v10: 降低通道数(CUDA 56→48, NPU 96→64), 减少D过拟合
- EMA: 指数移动平均 (Generator 权重平滑)

v12 核心改动 (2026-08) — 修复 z 完全失效导致的同标签模式坍塌:
  根因分析 (基于 v11 300轮实测):
    1. CondUpBlock 噪声注入被 `if self.training` 门控, eval 模式(推理)完全禁用
    2. NOISE_STD=0.02 固定常数, 相对特征std≈1 仅 2%, 且不可学习
    3. 最后一层 up3 无任何噪声/FiLM注入, 高分辨率细节与 z 无关
    4. z 输入梯度 0.047 vs text 梯度 10.75 (200倍差距) — 网络学会忽略 z
  修复:
  - [P0] CondUpBlock 噪声改为 per-channel 可学习增益 (StyleGAN式),
        初始 0.05, 训练/推理均注入 — 同标签不同z采样天然多样
  - [P0] up3 增加 NoiseInjection (高分辨率细节也依赖随机性)
  - [P0] 配合 train.py 同标签双z对比多样性损失, 给 z 路径梯度压力
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from config import *


# ═══════════════════════════════════════════════════════════
#  噪声注入 (v12: StyleGAN式 per-channel 可学习增益)
# ═══════════════════════════════════════════════════════════

class NoiseInjection(nn.Module):
    """
    v12: per-channel 可学习噪声注入 (StyleGAN式).
    训练和推理均注入随机噪声, 增益可学习 — 让同标签不同z采样天然多样.

    相比 v11 的固定 NOISE_STD:
      - 每通道独立增益 (StyleGAN 用 zeros 初始化, 此处用 0.05 保证初期即有随机性)
      - 训练/推理一致生效 (v11 的 `if self.training` 门控导致 eval 零多样性)
    """

    def __init__(self, channels, init_std=None):
        super().__init__()
        if init_std is None:
            init_std = NOISE_STD
        # v12: 初始 0.05, 可学习 (梯度会推动增益变大以满足多样性损失)
        self.gain = nn.Parameter(torch.full((channels,), float(init_std)))

    def forward(self, x):
        noise = torch.empty(x.shape, device=x.device, dtype=x.dtype).normal_(0, 1)
        return x + noise * self.gain.view(1, -1, 1, 1)


# ═══════════════════════════════════════════════════════════
#  Self-Attention
# ═══════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    """轻量 Self-Attention: 帮助 Generator 捕捉字符之间的空间关系"""

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


class ConditionalEmbedding(nn.Module):
    """文本条件编码 (保留以备将来使用)"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.fc = nn.Sequential(
            nn.Linear(EMBED_DIM * CAPTCHA_LENGTH, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, labels):
        B = labels.shape[0]
        e = self.embed(labels).view(B, -1)
        return self.fc(e)


# ═══════════════════════════════════════════════════════════
#  Generator v10: 渐进式生成 + FiLM条件注入 + 噪声
# ═══════════════════════════════════════════════════════════

class CondUpBlock(nn.Module):
    """
    v10: 带FiLM条件注入的上采样块.
    每个块: Upsample → CondConv → BN → ReLU → Noise
    文本条件通过scale+bias注入每一层, 让字符信息在各级分辨率生效.
    v12: 噪声注入改为 per-channel 可学习增益, 训练/推理均生效.
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

        # FiLM: 条件投影到 per-channel scale + bias
        self.cond_scale = nn.Linear(cond_dim, out_ch)
        self.cond_bias = nn.Linear(cond_dim, out_ch)

        # v12: per-channel 可学习噪声增益 (替代固定 NOISE_STD)
        # 初始 0.05, 由多样性损失推动增长, 训练/推理均注入
        self.noise_gain = nn.Parameter(torch.full((out_ch,), float(noise_std)))

    def forward(self, x, cond):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)

        # FiLM调制: scale=1+α, bias=β
        scale = self.cond_scale(cond).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        bias = self.cond_bias(cond).unsqueeze(-1).unsqueeze(-1)
        x = x * (1.0 + scale) + bias
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        # v12: 噪声注入始终生效 (训练+推理), 增益可学习
        # v11 的 `if self.training` 门控导致 eval 模式零多样性 — 已移除
        noise = torch.empty(x.shape, device=x.device, dtype=x.dtype).normal_(0, 1)
        x = x + noise * self.noise_gain.view(1, -1, 1, 1)
        return x


class Generator(nn.Module):
    """
    v10 渐进式条件生成器:
    - 输入: 噪声 z (B, 256) + 文本条件 (B, 256)
    - 输出: 90×35 灰度图 (B, 1, 35, 90)

    v10 核心改进 (基于v9 236轮诊断):
    - 废除 init_proj (512→24576 单层, 12.5M params),
      改为小初始化 Linear(512→6144) → 4×6 spatial.
    - CondUpBlock ×3: 每级注入FiLM条件, 字符信息在各级分辨率生效.
    - Self-Attention 在最低分辨率 (4×6) 捕获全局结构.
    - 无Skip Connections: 避免Encoder过拟合特征被Decoder直接复用.
    - 参数量 ~6M (vs v9的12.6M+), 减少50%+.
    """

    def __init__(self):
        super().__init__()
        self.text_embed = TextEmbedding()
        C = 64  # v10: base channels (up from 32, but fewer blocks so similar total)

        # ── 噪声+文本 → 共享条件 ──
        self.cond_proj = nn.Sequential(
            nn.Linear(LATENT_DIM + 256, 512),
            nn.ReLU(inplace=True),
        )

        # ── 小初始化: 512 → (C*4) × 4 × 6 = 256 × 24 = 6144 ──
        # 仅 ~3.2M 参数 (vs v9 init_proj的12.5M), 减少75%
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
        self.attn = SelfAttention(init_ch)  # 4×6 → 24 tokens, 很高效的attention

        # ── 渐进式上采样 (FiLM注入在每级) ──
        # 4×6 → 8×12 → 16×24 → 35×90
        self.up1 = CondUpBlock(init_ch, C * 2, cond_dim=512,
                               scale_h=2, scale_w=2)      # 4×6 → 8×12
        self.up2 = CondUpBlock(C * 2, C, cond_dim=512,
                               scale_h=2, scale_w=2)      # 8×12 → 16×24
        # Final block: 16×24 → 35×90 (non-uniform upsampling)
        # v12: 增加 NoiseInjection — 高分辨率细节也依赖随机性 (v11 此块无任何注入)
        self.up3 = nn.Sequential(
            nn.Upsample(size=(35, 90), mode='bilinear', align_corners=False),
            nn.Conv2d(C, C // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 2), nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 4), nn.ReLU(inplace=True),
            NoiseInjection(C // 4),
        )

        # ── Final output ──
        self.final = nn.Conv2d(C // 4, 1, 3, padding=1, bias=False)

    def forward(self, z, labels):
        B = z.size(0)
        init_ch = 256  # C * 4 = 64 * 4

        # 1. 条件编码
        cond = self.text_embed(labels)                     # (B, 256)
        cond = self.cond_proj(torch.cat([z, cond], dim=1))  # (B, 512)

        # 2. 小初始化 → 最小空间特征
        x = self.init_fc(cond)                              # (B, 6144)
        x = x.view(B, init_ch, self.init_h, self.init_w)    # (B, 256, 4, 6)

        # 3. Bottleneck + Self-Attention
        x = self.bottleneck(x)
        x = self.attn(x)

        # 4. 渐进式上采样 + FiLM条件注入
        x = self.up1(x, cond)           # (B, 128, 8, 12)
        x = self.up2(x, cond)           # (B, 64, 16, 24)
        x = self.up3(x)                 # (B, 16, 35, 90)

        # 5. Final → Tanh
        x = self.final(x)               # (B, 1, 35, 90)
        return torch.tanh(x).contiguous()


# ═══════════════════════════════════════════════════════════
#  Discriminator (AC-GAN + SpectralNorm + 单共享池化)
# ═══════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """
    条件判别器 (v10 Per-Position Aux + Minibatch StdDev, 降低容量):

    v10: DIS_FEATURES 56→48 (CUDA), 96→64 (NPU), 减少D过拟合训练集.

    v8 Per-Position Aux heads: 将特征图空间拆为4列, 每列对应一个字符位置,
    各有独立的轻量分类器(256→36), 强制空间分离字符 — 直接解决粘连.

    v6 Minibatch StdDev: 帮助 D 检测模式坍塌.
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

        # v6: Minibatch StdDev 在 conv4 之后、pool 之前注入, +1 通道
        # v3 NPU 修复: AdaptiveAvgPool2d((2,4)) → AvgPool2d((1,2), stride=1)
        # 输入 (B, C+1, 2, 5) → 输出 (B, C+1, 2, 4), 避免 NPU backward 维度不匹配
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), stride=1)
        # v6: +1 for minibatch stddev channel
        self._num_spatial_ch = DIS_FEATURES * 8 + 1  # v8: 空间通道数 (for per-position Aux)
        self._spatial_h = 2
        self._spatial_w = 4  # v8: 4个宽度位置 ← 对应4个字符位置
        pooled_dim = self._num_spatial_ch * self._spatial_h * self._spatial_w  # e.g. (56*8+1)*2*4=3592 (CUDA)

        # 判别头 (真/假)
        self.disc_head = nn.Sequential(
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(pooled_dim, 1, bias=False)),
        )

        # v8: Per-position Aux heads — 防字符粘连
        # 将 AvgPool2d 后的特征图 (B, C+1, 2, 4) 按宽度拆为4列,
        # 每列对应一个字符位置, 各有独立的小分类器。
        # 这强制 D 从不同空间区域识别不同位置的字符,
        # 进而推动 G 在空间上分离字符 (直接解决粘连问题)。
        pos_dim = self._num_spatial_ch * self._spatial_h  # (C+1) * 2 = per-position feature dim
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
        """统一 backbone: 可选收集中间层特征

        v6: conv4 之后注入 Minibatch StdDev 通道,
        帮助 D 检测模式坍塌 (所有假图 std≈0 → 坍塌信号)。
        参考: Progressive GANs (Karras et al. 2018)
        """
        feats = [] if collect_features else None

        x = self.conv1(x); x = self.in1(x); x = self.act1(x)
        if collect_features: feats.append(x)

        x = self.conv2(x); x = self.in2(x); x = self.act2(x)
        if collect_features: feats.append(x)

        x = self.conv3(x); x = self.in3(x); x = self.act3(x)
        if collect_features: feats.append(x)

        x = self.conv4(x); x = self.in4(x); x = self.act4(x)
        if collect_features: feats.append(x)

        # --- v6: Minibatch StdDev (Progressive GANs) ---
        # 在 pool 之前注入, 作为额外特征通道
        # std over batch → (1, 1, H, W) → expand to (B, 1, H, W)
        B, C, H, W = x.shape
        mb_std = x.std(dim=0, keepdim=True).mean(dim=(1), keepdim=True)  # (1, 1, H, W)
        mb_std = mb_std.expand(B, 1, H, W)  # (B, 1, H, W)
        x = torch.cat([x, mb_std], dim=1)  # (B, C+1, H, W)

        # v3: AvgPool2d((1,2), stride=1) 替代 AdaptiveAvgPool2d((2,4))
        h = self.pool(x)
        # v3: .contiguous().reshape() 替代 nn.Flatten(),
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

        # v8: Per-position Aux — 将特征图空间拆分为4个位置, 每个位置独立分类
        # h shape: (B, pooled_dim) = (B, (C+1)*2*4)
        # Reshape → (B, C+1, 2, 4) → split width → 4×(B, (C+1)*2)
        # 每个位置独立分类, 强制 D 关注不同空间区域, 防止字符粘连
        h_spatial = h.reshape(h.size(0), self._num_spatial_ch,
                              self._spatial_h, self._spatial_w)

        # v7: ReACGAN hypersphere 投影 (v8: per-position, 各自归一化)
        # 参考: Kang et al., "Rebooting ACGAN", NeurIPS 2021
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

    def get_features_only(self, x):
        """仅提取中间层特征 (用于 Feature Matching Loss)"""
        _, feats = self._backbone(x, collect_features=True)
        return feats


# ═══════════════════════════════════════════════════════════
#  EMA (指数移动平均)
# ═══════════════════════════════════════════════════════════

class EMAGenerator:
    """Generator EMA 副本, 用于采样和保存

    v5 修复 (2026-07):
      - update() 同时同步 BN buffers (running_mean, running_var),
        修复 EMA 采样时 BN 统计量冻结在初始化值 (mean=0, var=1) 的严重 Bug。
        此前所有 save_sample 输出的样本图均使用了错误的 BN 统计量,
        导致 EMA 采样质量远差于训练中 G 的实际输出。
    """

    def __init__(self, generator, decay=EMA_DECAY):
        self.ema_model = copy.deepcopy(generator).eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad = False
        # v5: 同时冻结 buffers 的梯度 (BN running_mean/var 不需要梯度)
        for b in self.ema_model.buffers():
            b.requires_grad = False

    def update(self, generator):
        """EMA 更新: 同步可训练参数 + BN 统计量 (buffers)

        v5 修复: parameters() 只包含可训练权重 (Conv weight/bias),
        不包含 BN 的 running_mean/running_var (注册为 buffers)。
        此前 EMA 的 BN 统计量永远停留在初始化值 (mean=0, var=1),
        导致 eval 模式下 BN 归一化完全错误, 所有样本图不可信。
        """
        with torch.no_grad():
            # 同步可训练参数 (EMA 平滑)
            for ema_p, g_p in zip(self.ema_model.parameters(), generator.parameters()):
                ema_p.data.mul_(self.decay).add_(g_p.data, alpha=1 - self.decay)
            # v5: 同步 BN 统计量 (直接复制训练中的 running_mean/var)
            # buffers 不需要 EMA 平滑 — 直接使用训练 G 的最新统计量
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
    """高斯初始化"""
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
    """正交初始化 (适用于 SpectralNorm + WGAN-GP)"""
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
#  测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from device_utils import DEVICE
    device = DEVICE

    G = Generator().to(device).apply(weights_init)
    D = Discriminator().to(device).apply(weights_init)

    z = torch.randn(4, LATENT_DIM).to(device)
    labels = torch.randint(0, NUM_CLASSES, (4, CAPTCHA_LENGTH)).to(device)

    fake = G(z, labels)
    print(f"Generator v10 output: {fake.shape}  [{fake.min():.2f}, {fake.max():.2f}]")

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
    print(f"\nParams — Generator v10: {param_g/1e6:.2f}M, Discriminator: {param_d/1e6:.2f}M")
