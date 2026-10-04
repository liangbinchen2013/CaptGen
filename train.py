"""
验证码生成器训练入口

模型: AC-GAN (条件生成器 + 判别器 + 逐位辅助分类), Hinge 对抗损失,
      R1/R2 梯度惩罚, StyleGAN2-ADA 自适应判别器增强, EMA 权重平滑。

训练阶段
  1. D 预训练        判别器先学会区分真假, 建立基础判别力
  2. G 预热          生成器仅用辅助分类/边缘/分布等损失学习字形
  3. 渐进对抗        对抗权重线性爬升, 联合优化 G 与 D
  4. OCR 课程学习    前期以 CaptchaResNet 强引导字形, 后期三 OCR 防欺骗
  5. Phase C         以真实机器混淆画像软化监督目标 (机器干扰匹配)

用法:
  python train.py                                            # 使用默认配置
  python train.py --data-dir ./训练数据                       # 指定数据路径
  python train.py --output-dir ./output                      # 指定输出路径
  python train.py --resume ./output/checkpoint_epoch_100.pt  # 断点续训
  python train.py --epochs 300 --batch-size 256              # 自定义训练参数
"""
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from config import *
from models import (
    Generator, Discriminator, weights_init, weights_init_ortho,
    EMAGenerator,
)
import ocr_net

_CONTRAST_MARGIN = CONTRAST_MARGIN
_CONTRAST_LO_Q = CONTRAST_LO_Q
_CONTRAST_HI_Q = CONTRAST_HI_Q
_D_UPDATES_MAX = D_UPDATES_MAX
_D_ADAPTIVE_THRESHOLD = D_ADAPTIVE_THRESHOLD
from data_loader import get_dataloader
from utils import save_sample, load_model
from device_utils import (
    DEVICE, device_type, device_supports_pin_memory,
    empty_cache, setup_matmul_precision,
    get_scaler, autocast, scaler_unscale_, use_amp,
    print_npu_env_hints,
)

# ─── 日志配置 ────────────────────────────────────────────
_FILE_LOG_INTERVAL = 100   # CSV 文件日志间隔 (步数), 比控制台更详细


# ═══════════════════════════════════════════════════════════
#  Hinge Loss (全设备统一, 替代 WGAN-GP)
# ═══════════════════════════════════════════════════════════
# WGAN-GP 不约束 D 的绝对值范围, 训练中 score 可漂移, 导致 G 梯度退化;
# Hinge Loss 迫使 D 在 [-1, 1] 区间工作, 配合 SpectralNorm 约束 Lipschitz,
# 无需额外梯度惩罚 (参考 SNGAN + Hinge, Miyato 2018), D_UPDATES=1 即可稳定。

_D_UPDATES_HINGE = D_UPDATES


def get_adaptive_d_updates(d_real_ema_val: float, d_fake_ema_val: float) -> int:
    """
    根据判别间隙动态调整 D 更新次数:
      - Gap > 1.0:        D 健康, 基础 1 次更新
      - Gap 0.3-1.0:      D 偏弱, 增加到 2 次
      - Gap < 0.3:        D 严重落后, 增加到 3 次 (上限)
    防止 D 被 G 碾压后无法恢复。
    """
    gap = d_real_ema_val - d_fake_ema_val
    if gap > 1.0:
        return 1
    elif gap > _D_ADAPTIVE_THRESHOLD:
        return 2
    else:
        return _D_UPDATES_MAX  # 3

print(f"  [LOSS] Hinge Loss + R1+R2 梯度惩罚 + ADA增强, D 每步更新 {_D_UPDATES_HINGE} 次")


# ═══════════════════════════════════════════════════════════
#  ADA (自适应判别器增强, 参考 StyleGAN2-ADA)
# ═══════════════════════════════════════════════════════════
# 对 D 输入应用随机增强, 防止 D 过拟合有限训练数据。
# 增强概率 p 自适应调整: D 过拟合 → p↑, D 欠拟合 → p↓。
# 验证码安全增强 (排除水平翻转, 避免改变字符顺序):
# 平移 (≤2px) / 缩放 (0.95-1.05) / 亮度对比度 / 高斯噪声 / Cutout。
# 注意: 增强仅应用于 D 输入, G 始终看到干净图像。


class ADAAugment:
    """自适应判别器增强 (StyleGAN2-ADA 简化版, 适配灰度验证码)"""

    def __init__(self, target: float = 0.6, p_inc: float = 2e-4,
                 p_dec: float = 5e-5, adjust_interval: int = 4):
        self.p = 0.0                     # 当前增强概率
        self.target = target             # D 对真实数据正确率目标
        self.p_inc = p_inc               # 概率增量
        self.p_dec = p_dec               # 概率减量
        self.adjust_interval = adjust_interval
        self._step = 0
        self._accum_real_sign = 0.0      # 累积 D 对真实数据的输出符号
        self._accum_count = 0

    def update_p(self, d_real_mean: float) -> float:
        """每 adjust_interval 步更新 p, 返回当前 p"""
        # d_real_mean > 0 表示 Hinge 下分类正确
        self._accum_real_sign += float(d_real_mean > 0)
        self._accum_count += 1
        self._step += 1

        if self._step % self.adjust_interval == 0 and self._accum_count > 0:
            r = self._accum_real_sign / self._accum_count  # 当前正确率
            if r > self.target:
                self.p += self.p_inc   # D 过拟合, 增强概率↑
            else:
                self.p -= self.p_dec   # D 欠拟合, 增强概率↓
            self.p = max(0.0, min(1.0, self.p))
            self._accum_real_sign = 0.0
            self._accum_count = 0
        return self.p

    @staticmethod
    def augment(images: "torch.Tensor", p: float) -> "torch.Tensor":
        """
        对 D 输入 batch 应用随机增强 (验证码安全集合)。
        images: (B, 1, H, W) in tanh [-1, 1], 返回同 shape 同 range。
        """
        import random as _random
        device = images.device
        aug = images.clone()

        # 1. 平移 (≤2px, 不改变字符顺序)
        if _random.random() < p:
            shift_x = _random.randint(-2, 2)
            shift_y = _random.randint(-2, 2)
            if shift_x != 0 or shift_y != 0:
                aug = torch.roll(aug, shifts=(shift_y, shift_x), dims=(-2, -1))

        # 2. 缩放 (0.95-1.05, 轻微)
        if _random.random() < p * 0.5:
            scale = 0.95 + _random.random() * 0.1  # [0.95, 1.05]
            aug = F.interpolate(aug, scale_factor=scale, mode='bilinear',
                                align_corners=False)
            # 裁剪/填充回原始尺寸
            H, W = aug.shape[2], aug.shape[3]
            if H > images.shape[2]:
                aug = aug[:, :, :images.shape[2], :]
            elif H < images.shape[2]:
                pad = images.shape[2] - H
                aug = F.pad(aug, (0, 0, pad // 2, pad - pad // 2), value=-1.0)
            if W > images.shape[3]:
                aug = aug[:, :, :, :images.shape[3]]
            elif W < images.shape[3]:
                pad = images.shape[3] - W
                aug = F.pad(aug, (pad // 2, pad - pad // 2, 0, 0), value=-1.0)

        # 3. 亮度/对比度 (灰度图安全)
        if _random.random() < p * 0.5:
            brightness = 0.85 + _random.random() * 0.3  # [0.85, 1.15]
            contrast = 0.9 + _random.random() * 0.2     # [0.9, 1.1]
            mean_val = aug.mean(dim=(2, 3), keepdim=True)
            aug = contrast * (aug - mean_val) + mean_val * brightness
            aug = aug.clamp(-1.0, 1.0)

        # 4. 高斯噪声
        if _random.random() < p * 0.3:
            noise_std = 0.02 * _random.random()  # [0, 0.02] in tanh space
            noise = torch.randn_like(aug) * noise_std
            aug = aug + noise
            aug = aug.clamp(-1.0, 1.0)

        # 5. Cutout (随机遮挡)
        if _random.random() < p * 0.3:
            H_img, W_img = aug.shape[2], aug.shape[3]
            cut_h = max(2, int(H_img * 0.15 * _random.random()))  # 0-15% height
            cut_w = max(3, int(W_img * 0.15 * _random.random()))  # 0-15% width
            cut_y = _random.randint(0, H_img - cut_h)
            cut_x = _random.randint(0, W_img - cut_w)
            aug[:, :, cut_y:cut_y + cut_h, cut_x:cut_x + cut_w] = -1.0

        return aug


# ═══════════════════════════════════════════════════════════
#  对抗损失权重调度
# ═══════════════════════════════════════════════════════════

def get_adv_lambda(epoch: int, warmup_epochs: int) -> float:
    """
    对抗损失权重调度:
      G 预热期:                        0.0 (纯辅助损失)
      warmup_epochs+1 — +ADV_WARMUP:  线性增长
      > warmup_epochs+ADV_WARMUP:      ADV_LAMBDA_MAX
    对抗仅作微调, 防止压倒辅助与视觉质量信号。
    """
    adv_warmup_epochs = ADV_WARMUP_EPOCHS
    if epoch <= warmup_epochs:
        return 0.0
    elif epoch <= warmup_epochs + adv_warmup_epochs:
        raw = (epoch - warmup_epochs) / adv_warmup_epochs
        return raw * ADV_LAMBDA_MAX
    else:
        return float(ADV_LAMBDA_MAX)


def get_r1_lambda(global_step: int) -> float:
    """R1 权重在训练前 5000 步从 0 线性增长, 避免初期惩罚过大导致 NaN"""
    r1_warmup_steps = 5000
    if global_step >= r1_warmup_steps:
        return float(LAMBDA_R1)
    return float(LAMBDA_R1) * (global_step / r1_warmup_steps)


def get_r2_lambda(global_step: int) -> float:
    """R2 权重与 R1 对称增长 (R1+R2 需同步引入以维持梯度平衡)"""
    r2_warmup_steps = 5000
    if global_step >= r2_warmup_steps:
        return float(LAMBDA_R2)
    return float(LAMBDA_R2) * (global_step / r2_warmup_steps)


def compute_aux_accuracy(aux_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    逐字符准确率 (0-dim tensor, 不触发设备同步)。
    aux_logits: (B, CAPTCHA_LENGTH, NUM_CLASSES); labels: (B, CAPTCHA_LENGTH)
    """
    preds = aux_logits.argmax(dim=-1)
    return (preds == labels).float().mean()


def compute_perfect_accuracy(aux_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """整串正确率: 4 个字符全部正确才算对 (0-dim tensor)"""
    preds = aux_logits.argmax(dim=-1)
    return (preds == labels).all(dim=1).float().mean()


def diversity_loss(fake_imgs: torch.Tensor, fake_imgs2: torch.Tensor,
                   threshold: float = 0.4) -> torch.Tensor:
    """
    同标签双噪声对比多样性损失 — 直接监督噪声路径的使用。

    对同一批标签用 z1/z2 各生成一张, 计算两图 per-pixel L1 距离,
    低于阈值则惩罚: 要想损失为 0, 噪声必须真实影响输出。
    避免生成器退化为"标签→图像"的确定性函数 (同标签采样趋同)。

    Args:
        fake_imgs:  G(z1, labels), tanh [-1,1]
        fake_imgs2: G(z2, labels), 同一批 labels, tanh [-1,1]
        threshold:  目标最小 per-pixel L1 距离 (tanh 空间)
    """
    dist = (fake_imgs - fake_imgs2).abs().mean(dim=(1, 2, 3))  # (B,)
    return F.relu(threshold - dist).mean()


def edge_loss(fake_imgs: torch.Tensor, real_imgs: torch.Tensor) -> torch.Tensor:
    """
    Sobel 边缘 L1 匹配 — 直接约束生成锐利字符笔画。

    对 fake 和 real 分别提取 Sobel 边缘幅度, 匹配两者的:
    1. 边缘均值 (平均边缘强度)
    2. 边缘稀疏度 (锐利边缘=少数大值, 模糊边缘=均匀小值)
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=fake_imgs.dtype, device=fake_imgs.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=fake_imgs.dtype, device=fake_imgs.device).view(1, 1, 3, 3)

    gx_f = F.conv2d(fake_imgs, sobel_x, padding=1)
    gy_f = F.conv2d(fake_imgs, sobel_y, padding=1)
    edge_f = torch.sqrt(gx_f ** 2 + gy_f ** 2 + 1e-8)

    gx_r = F.conv2d(real_imgs, sobel_x, padding=1)
    gy_r = F.conv2d(real_imgs, sobel_y, padding=1)
    edge_r = torch.sqrt(gx_r ** 2 + gy_r ** 2 + 1e-8)

    edge_mean_loss = F.l1_loss(edge_f.mean(), edge_r.mean())
    fake_edge_ratio = (edge_f > 0.05).float().mean()
    real_edge_ratio = (edge_r > 0.05).float().mean()
    edge_sparsity_loss = F.l1_loss(fake_edge_ratio, real_edge_ratio)

    return edge_mean_loss + 0.5 * edge_sparsity_loss


def realism_loss(fake_imgs: torch.Tensor, real_imgs: torch.Tensor) -> torch.Tensor:
    """
    分布统计量匹配 — 批量级约束生成图逼近真实数据的视觉统计特性。

    匹配三个维度:
    1. 逐图像素均值 (亮度分布)
    2. 逐图像素标准差 (对比度分布)
    3. 逐图梯度幅度 (纹理复杂度)
    """
    B = fake_imgs.size(0)
    fake_flat = fake_imgs.view(B, -1)
    real_flat = real_imgs.view(B, -1)

    fake_mean = fake_flat.mean(dim=1)
    real_mean = real_flat.mean(dim=1)
    mean_loss = F.l1_loss(fake_mean.mean(), real_mean.mean()) + \
                F.l1_loss(fake_mean.std(), real_mean.std())

    fake_std = fake_flat.std(dim=1)
    real_std = real_flat.std(dim=1)
    std_loss = F.l1_loss(fake_std.mean(), real_std.mean()) + \
               F.l1_loss(fake_std.std(), real_std.std())

    dy_f = fake_imgs[:, :, 1:, :] - fake_imgs[:, :, :-1, :]
    dx_f = fake_imgs[:, :, :, 1:] - fake_imgs[:, :, :, :-1]
    grad_f = torch.cat([dy_f.abs().mean(dim=(1, 2, 3)),
                        dx_f.abs().mean(dim=(1, 2, 3))], dim=0)

    dy_r = real_imgs[:, :, 1:, :] - real_imgs[:, :, :-1, :]
    dx_r = real_imgs[:, :, :, 1:] - real_imgs[:, :, :, :-1]
    grad_r = torch.cat([dy_r.abs().mean(dim=(1, 2, 3)),
                        dx_r.abs().mean(dim=(1, 2, 3))], dim=0)

    grad_loss = F.l1_loss(grad_f.mean(), grad_r.mean())

    return mean_loss + std_loss + grad_loss


# ═══════════════════════════════════════════════════════════
#  反粘连 gap 墨迹损失 + Phase C 混淆软目标
# ═══════════════════════════════════════════════════════════

def gap_ink_loss(fake_imgs: torch.Tensor, target_max: float = None,
                 tau: float = None, k: float = 20.0) -> torch.Tensor:
    """
    反粘连 gap 墨迹 hinge 损失 (可选, 默认关闭)。

    软 gap 比 = 3 条列界带平均墨迹密度 / 全图列墨迹峰值;
    仅当比值 > target_max 时惩罚 relu(ratio-target), 低处不奖励 —
    避免把生成逼离真实数据的轻微粘连分布 (无下界最小化会切出硬间隙、截断笔画)。
    默认 LAMBDA_GAP=0, 实验时用 --lambda-gap 开启。
    """
    if target_max is None:
        target_max = GAP_RATIO_MAX
    if tau is None:
        tau = GAP_INK_TAU
    g = (fake_imgs + 1.0) * 0.5                              # tanh→[0,1] 灰度
    B = g.size(0)
    bg = g.view(B, -1).median(dim=1).values.detach()         # 每图背景
    # 软墨迹 = softplus(k·(|g-bg|-tau))/k ≈ relu, 厚墨迹也有梯度;
    # 并扣除背景地板, 避免空白图 peak≈floor 造成 ratio=1 的退化值
    z = (g - bg.view(B, 1, 1, 1)).abs() - tau
    ink0 = F.softplus(torch.tensor(-k * tau, dtype=g.dtype, device=g.device)) / k
    ink = (F.softplus(k * z) / k - ink0).clamp(min=0)
    prof = ink.mean(dim=(0, 1, 2))                           # (W,) 列墨迹剖面
    peak = prof.detach().max().clamp(min=1e-6)               # 分母 stop-grad
    ratio = torch.stack([prof[x0:x1].mean()
                         for x0, x1 in GAP_STRIPS]).mean() / peak
    return F.relu(ratio - float(target_max))


def gap_ink_ratio(fake_imgs: torch.Tensor, tau: float = None) -> float:
    """
    gap 墨迹比 (日志指标, 与 confusion_profile.json 的 real_gap 同口径硬判定)。
    = 3 条列界带平均墨迹密度 / 全图列墨迹峰值。真实 ≈ 0.84/0.96/0.69。
    """
    if tau is None:
        tau = GAP_INK_TAU
    g = ((fake_imgs.detach() + 1.0) * 0.5 * 255.0)           # [0,255] 与实测同尺度
    B = g.size(0)
    bg = g.view(B, -1).median(dim=1).values.view(B, 1, 1, 1)
    ink = ((g - bg).abs() > tau * 255.0).float()             # 硬判定
    prof = ink.mean(dim=(0, 1, 2))
    peak = prof.max().clamp(min=1e-6)
    ratio = torch.stack([prof[x0:x1].mean()
                         for x0, x1 in GAP_STRIPS]).mean() / peak
    return float(ratio)


def build_soft_targets():
    """
    构造 Phase C 混淆软目标:
        y_soft[c] = (1-ε_c)·δ_c + ε_c·confusion_dist[c], ε_c = min(error_rate[c], EPS_CAP)
    Returns: dist_mat (36,36), eps (36,) — 缺失行 ε=0 退化为硬 CE。
    """
    dist_mat = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    for ch, row in CONFUSION_DIST.items():
        ci = CHARS.index(ch)
        for p, v in row.items():
            if p in CHARS:
                dist_mat[ci, CHARS.index(p)] = v
    eps = torch.zeros(NUM_CLASSES)
    for ch, v in OCR_ERROR_RATE.items():
        ci = CHARS.index(ch)
        # 无混淆分布数据的字符强制 ε=0 (保持归一化)
        if dist_mat[ci].sum() > 0.5:
            eps[ci] = min(float(v), EPS_CAP)
    return dist_mat, eps


def soft_ce(ocr_logits: torch.Tensor, labels: torch.Tensor,
            dist_mat: torch.Tensor, eps_vec: torch.Tensor,
            t: float) -> torch.Tensor:
    """
    Phase C 混淆软目标 CE。t=0 ≡ 硬 CE (连续过渡),
    t=1 → 目标为真实机器混淆画像。
    Returns: (B, CAPTCHA_LENGTH) 逐字符 CE。
    """
    B, L, C = ocr_logits.shape
    logp = F.log_softmax(ocr_logits.reshape(-1, C), dim=-1)   # (B*L, C)
    lab = labels.reshape(-1)
    onehot = F.one_hot(lab, C).float()
    eps = (eps_vec.to(lab.device)[lab] * float(t)).unsqueeze(1)   # (B*L,1)
    y = (1.0 - eps) * onehot + eps * dist_mat.to(lab.device)[lab]
    return -(y * logp).sum(-1).view(B, L)


_SOFT_DIST, _SOFT_EPS = build_soft_targets()


# ═══════════════════════════════════════════════════════════
#  梯度惩罚 (R1 / R2)
# ═══════════════════════════════════════════════════════════

def compute_r1_penalty(real_pred, real_img):
    """
    R1 梯度惩罚: E[||∇_x D(x)||²], 仅在真实数据上。
    强制 D 在真实数据周围保持平滑, 防止 Hinge margin 过冲。
    参考: Mescheder et al. 2018, StyleGAN 标配。
    """
    grad_real = autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True,
        only_inputs=True,
    )[0]
    return grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()


def compute_r2_penalty(fake_pred, fake_img):
    """
    R2 梯度惩罚: E[||∇_x D(x_fake)||²], 仅在生成数据上。
    与 R1 对称, R1+R2 联合强制 D 成为最大间隔分类器,
    防止 D 过拟合真实数据而忽略生成数据。
    参考: R3GAN (Huang et al., NeurIPS 2024)。
    """
    grad_fake = autograd.grad(
        outputs=fake_pred.sum(),
        inputs=fake_img,
        create_graph=True,
        only_inputs=True,
    )[0]
    return grad_fake.pow(2).reshape(grad_fake.shape[0], -1).sum(1).mean()


# ═══════════════════════════════════════════════════════════
#  特征匹配损失
# ═══════════════════════════════════════════════════════════

def feature_matching_loss(real_features, fake_features):
    """
    生成图在判别器中间层的特征应匹配真图的特征统计量 (逐层 L1)。
    参考: Salimans et al. 2016 "Improved Techniques for Training GANs"。
    """
    total = 0.0
    n = 0
    for rf, ff in zip(real_features, fake_features):
        total += F.l1_loss(ff, rf.detach())
        n += 1
    return total / max(n, 1)


# ═══════════════════════════════════════════════════════════
#  CSV 文件日志
# ═══════════════════════════════════════════════════════════

_FILE_LOG_HEADER = (
    "timestamp,epoch,step,phase,"
    "d_loss,g_loss,aux_loss,fm_loss,fm_lambda,contrast_loss,div_loss,edge_loss,realism_loss,"
    "ocr_loss,ocr_ce,ocr_feat,ocr_lambda,ocr_w_mean,ocr_valid_ratio,ocr_acc,"
    "dddd_acc,ppll_acc,gap_ink,"
    "ocr_phase,ocr_ramp,"
    "r1_penalty,r2_penalty,"
    "d_real_mean,d_fake_mean,d_gap,d_real_ema,d_fake_ema,"
    "d_grad_norm,g_grad_norm,"
    "lr_g,lr_d,"
    "aux_acc_real,aux_acc_fake,perfect_acc_real,perfect_acc_fake,"
    "steps_per_sec,batch_size\n"
)


def _open_log_file(log_path: Path, resume: bool = False) -> object:
    """
    打开日志文件: 续训则追加, 否则写入表头。
    续训但日志文件不存在/为空时仍写表头, 防止新目录产生无表头 CSV。
    """
    write_header = ((not resume) or (not log_path.exists())
                    or log_path.stat().st_size == 0)
    mode = "a" if (resume and not write_header) else "w"
    f = open(log_path, mode, encoding="utf-8", buffering=1)  # 行缓冲
    if write_header:
        f.write(_FILE_LOG_HEADER)
        f.flush()
    return f


def _write_log_line(f, timestamp: str, epoch: int, step: int, phase: str,
                    d_loss: float, g_loss: float, aux_loss: float,
                    fm_loss: float, fm_lambda: float, contrast_loss: float,
                    div_loss: float,
                    edge_loss_val: float,
                    realism_loss_val: float,
                    r1_penalty: float, r2_penalty: float,
                    d_real: float, d_fake: float,
                    d_real_ema: float, d_fake_ema: float,
                    d_grad: float, g_grad: float,
                    lr_g: float, lr_d: float,
                    aux_acc_real: float, aux_acc_fake: float,
                    perfect_acc_real: float = 0.0, perfect_acc_fake: float = 0.0,
                    ips: float = 0.0, batch_size: int = 0,
                    ocr_loss: float = 0.0, ocr_ce: float = 0.0,
                    ocr_feat: float = 0.0, ocr_lambda: float = 0.0,
                    ocr_w_mean: float = 0.0, ocr_valid_ratio: float = 0.0,
                    ocr_acc: float = 0.0,
                    dddd_acc: float = 0.0, ppll_acc: float = 0.0,
                    gap_ink: float = 0.0,
                    ocr_phase: str = "B", ocr_ramp: float = 1.0):
    """写入一行 CSV 日志"""
    line = (f"{timestamp},{epoch},{step},{phase},"
            f"{d_loss:.6f},{g_loss:.6f},{aux_loss:.6f},"
            f"{fm_loss:.6f},{fm_lambda:.4f},{contrast_loss:.6f},{div_loss:.6f},"
            f"{edge_loss_val:.6f},{realism_loss_val:.6f},"
            f"{ocr_loss:.6f},{ocr_ce:.6f},{ocr_feat:.6f},"
            f"{ocr_lambda:.4f},{ocr_w_mean:.4f},{ocr_valid_ratio:.4f},{ocr_acc:.4f},"
            f"{dddd_acc:.4f},{ppll_acc:.4f},{gap_ink:.4f},"
            f"{ocr_phase},{ocr_ramp:.2f},"
            f"{r1_penalty:.6f},{r2_penalty:.6f},"
            f"{d_real:+.4f},{d_fake:+.4f},{d_real - d_fake:+.4f},"
            f"{d_real_ema:+.4f},{d_fake_ema:+.4f},"
            f"{d_grad:.2f},{g_grad:.2f},"
            f"{lr_g:.6e},{lr_d:.6e},"
            f"{aux_acc_real:.4f},{aux_acc_fake:.4f},"
            f"{perfect_acc_real:.4f},{perfect_acc_fake:.4f},"
            f"{ips:.2f},{batch_size}\n")
    f.write(line)
    f.flush()


# ═══════════════════════════════════════════════════════════
#  梯度工具
# ═══════════════════════════════════════════════════════════

def _batch_quantile(t: torch.Tensor, q: float) -> torch.Tensor:
    """批内分位数 (NPU 不支持 torch.quantile 时用 sort 近似)"""
    try:
        return torch.quantile(t, q)
    except Exception:
        s = t.sort().values
        k = min(max(int(q * (s.numel() - 1)), 0), s.numel() - 1)
        return s[k]


def clip_gradients(model, max_norm=1.0):
    """裁剪模型梯度, 防止梯度爆炸"""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ═══════════════════════════════════════════════════════════
#  训练主循环
# ═══════════════════════════════════════════════════════════

def train(args):
    print(f"\n{'='*60}")
    print(f"  验证码 AC-GAN + Hinge Loss 训练")
    print(f"{'='*60}")
    print(f"  设备: {DEVICE} ({device_type})")
    print(f"  数据: {args.data_dir}")
    print(f"  输出: {args.output_dir}")
    print(f"  Batch: {args.batch_size}, Epochs: {args.epochs}")
    print(f"  学习率: G={args.lr_g}, D={args.lr_d}")
    print(f"  字符集: {NUM_CLASSES} 类")
    print(f"  损失权重: Aux={LAMBDA_AUX_G}, FM={LAMBDA_FM}, Edge={LAMBDA_EDGE}, Realism={LAMBDA_REALISM}, Adv<={ADV_LAMBDA_MAX}")
    print(f"  G 预热: {G_WARMUP_EPOCHS} epochs | 对抗渐进: {ADV_WARMUP_EPOCHS} epochs")
    print(f"  D 更新: {D_UPDATES} 次/步 (自适应, 最大{D_UPDATES_MAX}), D预训练: {args.d_pretrain} epochs")
    print(f"  EMA: decay={EMA_DECAY}")
    if device_type == "npu":
        print(f"  NPU 内部格式: 已禁用 (allow_internal_format=False)")
        print(f"  NPU R1 惩罚: {'禁用' if NPU_DISABLE_R1 else '启用'} (二阶梯度)")
        print(f"  NPU R2 惩罚: {'禁用' if NPU_DISABLE_R2 else '启用'} (二阶梯度)")
    if args.resume:
        print(f"  续训: {args.resume}")
    print(f"{'='*60}\n")

    # ─── 输出目录与文件日志 ────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = _open_log_file(output_dir / "training_log.csv",
                              resume=(args.resume is not None))

    # ─── 模型 ──────────────────────────────────────────
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    ema_g = EMAGenerator(G, decay=EMA_DECAY)
    ema_g.to(DEVICE)

    # ─── 优化器 ────────────────────────────────────────
    g_optim = optim.Adam(G.parameters(), lr=args.lr_g,
                         betas=(args.beta1, args.beta2))
    d_optim = optim.Adam(D.parameters(), lr=args.lr_d,
                         betas=(args.beta1, args.beta2))

    # ─── 混合精度 (NPU 用纯 fp32; CUDA 与 AM 二阶梯度不兼容, 同样 fp32) ──
    scaler = get_scaler() if use_amp() else None
    use_amp_flag = scaler is not None

    # ─── ADA 自适应判别器增强 ──────────────────────────
    if USE_ADA:
        ada = ADAAugment(target=ADA_TARGET, p_inc=ADA_P_INC,
                         p_dec=ADA_P_DEC, adjust_interval=ADA_INTERVAL)
        print(f"  [ADA] 已启用自适应判别器增强 (target={ADA_TARGET}, "
              f"interval={ADA_INTERVAL})")
    else:
        ada = None

    # ─── 学习率调度器 ──────────────────────────────────
    if hasattr(args, 'use_cosine_lr') and args.use_cosine_lr:
        lr_min = getattr(args, 'lr_min', 1e-5)
        g_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            g_optim, T_max=args.epochs, eta_min=lr_min)
        d_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            d_optim, T_max=args.epochs, eta_min=lr_min)
        print(f"  [LR] 使用Cosine Annealing调度器 (min_lr={lr_min})")
    else:
        g_scheduler = optim.lr_scheduler.MultiStepLR(
            g_optim, milestones=LR_MILESTONES, gamma=LR_DECAY_FACTOR)
        d_scheduler = optim.lr_scheduler.MultiStepLR(
            d_optim, milestones=LR_MILESTONES, gamma=LR_DECAY_FACTOR)

    # ─── 损失函数 ──────────────────────────────────────
    aux_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING_EPS)

    # ─── 断点续训 ──────────────────────────────────────
    start_epoch = 1
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        G.load_state_dict(ckpt["generator_state_dict"])
        D.load_state_dict(ckpt["discriminator_state_dict"])
        if "g_optimizer_state_dict" in ckpt and g_optim:
            g_optim.load_state_dict(ckpt["g_optimizer_state_dict"])
        if "d_optimizer_state_dict" in ckpt and d_optim:
            d_optim.load_state_dict(ckpt["d_optimizer_state_dict"])
        if "ema_state_dict" in ckpt:
            ema_g.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        # 恢复 LR scheduler 状态; 旧检查点无该字段时按 epoch 快进,
        # 防止余弦退火从 last_epoch=0 重启导致学习率跳回初始值。
        _sched_restored = False
        if "g_scheduler_state_dict" in ckpt and "d_scheduler_state_dict" in ckpt:
            try:
                g_scheduler.load_state_dict(ckpt["g_scheduler_state_dict"])
                d_scheduler.load_state_dict(ckpt["d_scheduler_state_dict"])
                _sched_restored = True
            except Exception as _e:
                print(f"[WARN] scheduler 状态恢复失败: {_e}")
        if not _sched_restored:
            _ff = max(start_epoch - 1, 0)
            if _ff > 0:
                # 快进仅调整 LR; 首次 optimizer.step() 前调用会触发 PyTorch
                # 提示, 此处为断点续训恢复场景, 安全忽略。
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", UserWarning)
                    for _ in range(_ff):
                        g_scheduler.step()
                        d_scheduler.step()
                print(f"[OK] 检查点无 scheduler 状态, 已快进 {_ff} 步 "
                      f"(LR={g_optim.param_groups[0]['lr']:.2e})")
        print(f"[OK] 从 epoch {start_epoch - 1} 恢复训练 (global_step={global_step})\n")
    else:
        G.apply(weights_init_ortho)
        D.apply(weights_init)
        ema_g.update(G)  # 初始 EMA 副本

    # ─── 数据加载 ──────────────────────────────────────
    loader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    pin_memory = device_supports_pin_memory()

    # ─── 固定噪声 (采样可视化用) ────────────────────────
    fixed_z = torch.randn(16, LATENT_DIM, device=DEVICE)
    fixed_labels = torch.randint(0, NUM_CLASSES, (16, CAPTCHA_LENGTH), device=DEVICE)
    # 第二组固定噪声: 对比 EMA 与训练中 G
    fixed_z2 = torch.randn(16, LATENT_DIM, device=DEVICE)
    fixed_labels2 = torch.randint(0, NUM_CLASSES, (16, CAPTCHA_LENGTH), device=DEVICE)

    # ─── 自适应训练状态 ────────────────────────────────
    d_real_ema = torch.tensor(0.0, device=DEVICE)   # tensor 跟踪, 避免 per-step 同步
    d_fake_ema = torch.tensor(0.0, device=DEVICE)
    _consecutive_nan = 0  # 连续 NaN 计数器, 超过阈值停止训练

    # ─── D 预训练: G 预热前先建立 D 基础判别力 ─────────
    if args.d_pretrain > 0 and not args.resume:
        print(f"\n  ═══ D 预训练阶段 ({args.d_pretrain} epochs): 建立D baseline判别力 ═══\n")
        d_pretrain_start = time.time()
        for dpe in range(1, args.d_pretrain + 1):
            dpe_loss = 0.0
            dpe_batches = 0
            for batch_idx, (real_imgs, labels) in enumerate(loader):
                B = real_imgs.size(0)
                real_imgs = real_imgs.to(DEVICE, non_blocking=pin_memory)
                labels = labels.to(DEVICE, non_blocking=pin_memory)

                d_optim.zero_grad()
                z = torch.randn(B, LATENT_DIM, device=DEVICE)
                with torch.no_grad():
                    fake_imgs = G(z, labels)

                real_imgs.requires_grad_(True)
                with autocast():
                    d_real, aux_real, _ = D(real_imgs, labels)
                    d_fake, aux_fake, _ = D(fake_imgs.detach(), labels)

                    d_loss_adv = (F.relu(1.0 + d_fake).mean() +
                                  F.relu(1.0 - d_real).mean())
                    aux_loss_real = aux_criterion(
                        aux_real.view(-1, NUM_CLASSES), labels.view(-1))
                    aux_loss_fake = aux_criterion(
                        aux_fake.view(-1, NUM_CLASSES), labels.view(-1))
                    aux_loss_avg = (aux_loss_real + aux_loss_fake) / 2
                    output_reg = 1e-4 * (d_real.mean() ** 2 + d_fake.mean() ** 2)

                    if LAMBDA_R1 > 0 and not NPU_DISABLE_R1:
                        r1_val = compute_r1_penalty(d_real.float(), real_imgs)
                    else:
                        r1_val = torch.tensor(0.0, device=DEVICE)

                    d_loss_total = (d_loss_adv + LAMBDA_AUX_D * aux_loss_avg +
                                    LAMBDA_R1 * r1_val + output_reg)

                if use_amp_flag:
                    scaler.scale(d_loss_total).backward()
                    scaler_unscale_(scaler, d_optim)
                else:
                    d_loss_total.backward()

                _ok = all(_p.grad is None or torch.isfinite(_p.grad).all()
                          for _p in D.parameters() if _p.grad is not None)
                if not _ok:
                    print(f"  [WARN] D pretrain grad NaN, skipping step")
                    d_optim.zero_grad()
                    if use_amp_flag:
                        scaler.update()
                    continue

                clip_gradients(D, CLIP_NORM)
                if use_amp_flag:
                    scaler.step(d_optim)
                    scaler.update()
                else:
                    d_optim.step()

                real_imgs.requires_grad_(False)
                dpe_loss += d_loss_total.detach()
                dpe_batches += 1

            avg_dpe = (dpe_loss / max(dpe_batches, 1)).item()
            print(f"  [D_PRETRAIN] Epoch {dpe}/{D_PRETRAIN_EPOCHS} | D Loss: {avg_dpe:.4f}")
        print(f"  ═══ D 预训练完成 ({time.time() - d_pretrain_start:.0f}s) ═══\n")

    # ─── OCR 课程学习设置 ──────────────────────────────
    use_curriculum = not args.no_curriculum
    cur_epochs = max(int(getattr(args, 'curriculum_epochs', CURRICULUM_EPOCHS)), 0)
    cur_ramp = max(int(getattr(args, 'curriculum_ramp', CURRICULUM_RAMP)), 0)
    _phase_c_epoch = max(int(getattr(args, 'phase_c_epoch', PHASE_C_EPOCHS)), 0)
    _phase_c_ramp = max(int(getattr(args, 'phase_c_ramp', PHASE_C_RAMP)), 0)

    def ocr_phase_of(ep: int):
        """纯 epoch 函数 → (phase, t); 断点续训自动正确"""
        return curriculum_phase(ep, cur_epochs, cur_ramp, use_curriculum)

    if use_curriculum:
        print(f"  [OCR] 课程学习: Phase A(Ep1-{cur_epochs}, 仅CaptchaResNet强引导) → "
              f"Ramp(Ep{cur_epochs+1}-{cur_epochs+cur_ramp}, 线性混合) → "
              f"Phase B(Ep{cur_epochs+cur_ramp+1}+, 三OCR防欺骗)")
    else:
        print(f"  [OCR] 课程学习已禁用 (--no-curriculum), 全程三OCR方案")
    if _phase_c_epoch > 0:
        print(f"  [OCR] Phase C(Ep{_phase_c_epoch}+): 混淆软目标 CE (ε爬升{_phase_c_ramp}ep) "
              f"— 机器干扰匹配 (ε_max={EPS_CAP})")

    # ─── 分层 OCR 自适应监督 ────────────────────────────
    # 1. ddddocr + ppllocr 联合检测 (2 个都超过阈值才算欺骗), 权重高
    # 2. CaptchaResNet 单独检测, 权重低, 同时提供 OCR 监督损失
    adaptive_ocr = None
    ocr_model = None  # CaptchaResNet, 用于 OCR 监督损失

    if LAMBDA_OCR > 0:
        from adaptive_ocr import MultiOCRAdaptive
        # Phase A 不加载第三方 OCR (省 CPU), Ramp 起点由 load_third_party() 幂等补载
        _ph0, _t0 = ocr_phase_of(start_epoch)
        adaptive_ocr = MultiOCRAdaptive(
            device=DEVICE, load_third_party=(not use_curriculum) or _t0 > 0)
        if 'captcha_resnet' in adaptive_ocr.ocr_models:
            ocr_model = adaptive_ocr.ocr_models['captcha_resnet']
        print(f"  [OCR] 启用分层OCR自适应监督")
        print(f"  [OCR] 初始权重: {adaptive_ocr.current_weight:.4f}")
        print(f"  [OCR] 已加载 {len(adaptive_ocr.ocr_models)} 个OCR模型")
        print(f"  [OCR] 第三方联合检测: ddddocr+ppllocr (2个都对才算欺骗)")
        print(f"  [OCR] CaptchaResNet单独检测 + OCR监督损失")
        for name, acc in adaptive_ocr.base_accuracies.items():
            print(f"  [OCR]   {name}: 基准{acc*100:.1f}%, 阈值{(acc+adaptive_ocr.cheat_threshold)*100:.1f}%")
    else:
        print(f"  [OCR] 已禁用 (LAMBDA_OCR=0.0), 使用纯视觉质量监督")

    def get_ocr_lambda(ep: int, batch_idx: int = 0, fake_imgs=None, labels=None) -> float:
        """获取当前 OCR 外层权重 (Phase A 固定, Phase B 自适应检测欺骗)"""
        if LAMBDA_OCR <= 0 or adaptive_ocr is None:
            return 0.0
        if ep <= G_WARMUP_EPOCHS:
            return 0.0

        # Phase A 含 Ramp 阶段固定权重, 不做欺骗检测 (第三方 OCR 可能未加载)
        _ph, _t = ocr_phase_of(ep)
        if _t < 1.0:
            ocr_weight = LAMBDA_OCR
        elif fake_imgs is not None and labels is not None:
            if batch_idx % 100 == 0:
                ocr_weight, accuracies, is_cheating, cheating_models = adaptive_ocr.check_and_update(
                    fake_imgs.detach(), labels
                )
                if batch_idx % 500 == 0:
                    print(f"  [OCR] {adaptive_ocr.get_summary()}")
            else:
                ocr_weight = adaptive_ocr.current_weight
        else:
            ocr_weight = adaptive_ocr.current_weight

        # 预热期渐进引入
        ramp = OCR_WARMUP_EPOCHS
        if ep <= G_WARMUP_EPOCHS + ramp:
            return ocr_weight * (ep - G_WARMUP_EPOCHS) / ramp
        return ocr_weight

    # ─── 训练 ──────────────────────────────────────────
    print("开始训练...\n")
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        in_g_warmup = (G_WARMUP_EPOCHS > 0 and epoch <= G_WARMUP_EPOCHS)
        in_adv_warmup = (not in_g_warmup and G_WARMUP_EPOCHS > 0 and
                         epoch <= G_WARMUP_EPOCHS + ADV_WARMUP_EPOCHS)
        # OCR 课程学习阶段 (纯 epoch 函数, 断点续训自动正确)
        ocr_ph, ocr_t = ocr_phase_of(epoch)
        # Phase C 混淆软目标 ε 爬升
        ocr_pc, ocr_pc_t = phase_c_t(epoch, _phase_c_epoch, _phase_c_ramp)
        if ocr_pc:
            ocr_ph = "C" if ocr_t >= 1.0 else ocr_ph   # 日志标签
        # ocr_ramp: Phase C(且已入 B) 记 ε 爬升系数, 否则记 A→B 混合系数
        ocr_ramp_v = float(ocr_pc_t) if (ocr_pc and ocr_t >= 1.0) else float(ocr_t)
        if ocr_t > 0 and adaptive_ocr is not None:
            # 幂等: Ramp 起点补载第三方 OCR (Phase A 未加载)
            adaptive_ocr.load_third_party()
        if epoch == 1 and in_g_warmup:
            print(f"  ═══ G 预热阶段 ({G_WARMUP_EPOCHS} epochs): G 仅用 Aux loss 训练 (无对抗损失) ═══\n")
        elif epoch == G_WARMUP_EPOCHS + 1 and G_WARMUP_EPOCHS > 0:
            empty_cache()
            print(f"\n  ═══ G 预热完成, 开始渐进对抗训练 (Hinge + Aux + FM + Contrast, {ADV_WARMUP_EPOCHS} epochs 渐进) ═══\n")

        epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_fm_loss = 0.0
        epoch_contrast_loss = 0.0
        epoch_div_loss = 0.0
        epoch_edge_loss = 0.0
        epoch_realism_loss = 0.0
        epoch_ocr_loss = 0.0
        epoch_ocr_ce = 0.0
        epoch_ocr_feat = 0.0
        epoch_ocr_lambda = 0.0
        epoch_ocr_w_mean = 0.0
        epoch_ocr_valid = 0.0
        epoch_ocr_acc = 0.0
        epoch_dddd_acc = 0.0
        epoch_ppll_acc = 0.0
        epoch_dddd_n = 0          # 实测样本计数 (统计均值时排除未测量 -1)
        epoch_ppll_n = 0
        epoch_gap_ink = 0.0
        epoch_aux_acc_real = 0.0
        epoch_aux_acc_fake = 0.0
        epoch_perfect_acc_real = 0.0
        epoch_perfect_acc_fake = 0.0
        epoch_batches = 0

        for batch_idx, (real_imgs, labels) in enumerate(loader):
            B = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE, non_blocking=pin_memory)
            labels = labels.to(DEVICE, non_blocking=pin_memory)

            # ─── D 更新频率 (根据判别间隙自适应) ──────────
            d_updates = get_adaptive_d_updates(
                d_real_ema.item(), d_fake_ema.item())

            # ═════════════════════════════════════════
            #  训练判别器 D
            # ═════════════════════════════════════════
            cached_real_features = None  # 缓存 D(real) 特征, 避免 G 步重复计算
            r1_penalty_val = torch.tensor(0.0, device=DEVICE)
            _skip_g_step = False         # D 梯度异常时跳过 G 训练
            _d_real_dstep = torch.tensor(0.0, device=DEVICE)
            _d_fake_dstep = torch.tensor(0.0, device=DEVICE)

            for d_iter in range(d_updates):
                d_optim.zero_grad()

                z = torch.randn(B, LATENT_DIM, device=DEVICE)
                with torch.no_grad():
                    fake_imgs = G(z, labels)

                need_features = (d_iter == d_updates - 1)  # 最后一次迭代收集特征
                need_r1 = (d_iter == d_updates - 1)

                if need_r1:
                    real_imgs.requires_grad_(True)

                with autocast():
                    # ADA 增强: 对 D 输入应用随机变换, 防止过拟合
                    if ada is not None:
                        _real_for_d = ada.augment(real_imgs, ada.p) if ada.p > 0 else real_imgs
                        _fake_for_d = ada.augment(fake_imgs.detach(), ada.p) if ada.p > 0 else fake_imgs.detach()
                    else:
                        _real_for_d = real_imgs
                        _fake_for_d = fake_imgs.detach()

                    if need_features:
                        d_real, aux_real, cached_real_features = D(_real_for_d, labels, return_features=True)
                    else:
                        d_real, aux_real, _ = D(_real_for_d, labels)
                    d_fake, aux_fake, _ = D(_fake_for_d, labels)

                    # Hinge 对抗损失
                    d_loss_adv = (F.relu(1.0 + d_fake).mean() +
                                  F.relu(1.0 - d_real).mean())

                    aux_loss_real = aux_criterion(
                        aux_real.view(-1, NUM_CLASSES), labels.view(-1))
                    aux_loss_fake = aux_criterion(
                        aux_fake.view(-1, NUM_CLASSES), labels.view(-1))
                    aux_loss_avg = (aux_loss_real + aux_loss_fake) / 2

                    # 输出弱正则化 (Hinge 天然限幅)
                    output_reg = 1e-4 * (d_real.mean() ** 2 + d_fake.mean() ** 2)

                # R1 梯度惩罚 (NPU 上默认禁用, 避免二阶梯度维度不匹配)
                if need_r1 and LAMBDA_R1 > 0 and not NPU_DISABLE_R1:
                    cur_r1_lambda = get_r1_lambda(global_step)
                    if cur_r1_lambda > 0:
                        r1_penalty_val = compute_r1_penalty(d_real.float(), real_imgs)
                    else:
                        r1_penalty_val = torch.tensor(0.0, device=DEVICE)
                else:
                    r1_penalty_val = torch.tensor(0.0, device=DEVICE)
                    cur_r1_lambda = 0.0

                # R2 梯度惩罚 (需 re-forward D(fake) 且 fake 带 grad)
                if need_r1 and LAMBDA_R2 > 0 and not NPU_DISABLE_R2:
                    cur_r2_lambda = get_r2_lambda(global_step)
                    if cur_r2_lambda > 0:
                        fake_imgs.requires_grad_(True)
                        with autocast():
                            d_fake_r2, _, _ = D(fake_imgs, labels)
                        r2_penalty_val = compute_r2_penalty(d_fake_r2.float(), fake_imgs)
                        fake_imgs.requires_grad_(False)
                    else:
                        r2_penalty_val = torch.tensor(0.0, device=DEVICE)
                else:
                    r2_penalty_val = torch.tensor(0.0, device=DEVICE)
                    cur_r2_lambda = 0.0

                d_loss_total = (d_loss_adv +
                                LAMBDA_AUX_D * aux_loss_avg +
                                cur_r1_lambda * r1_penalty_val +
                                cur_r2_lambda * r2_penalty_val +
                                output_reg)

                if use_amp_flag:
                    scaler.scale(d_loss_total).backward()
                    scaler_unscale_(scaler, d_optim)
                else:
                    d_loss_total.backward()

                # 梯度 NaN 检测, 防止污染模型权重
                _d_grad_ok = True
                for _name, _p in D.named_parameters():
                    if _p.grad is not None and (not torch.isfinite(_p.grad).all()):
                        _d_grad_ok = False
                        break
                if not _d_grad_ok:
                    _consecutive_nan += 1
                    if _consecutive_nan > 500:
                        print(f"  [FATAL] 连续 NaN 超过 500 次 (step {global_step}), 请降低学习率或增大 init_scale")
                        raise RuntimeError("训练崩溃: 连续 NaN 过多, 模型权重可能已损坏")
                    print(f"  [WARN] D 梯度 NaN at step {global_step} (连续{_consecutive_nan}次), 跳过本步 D+G 更新")
                    d_optim.zero_grad()
                    if use_amp_flag:
                        scaler.update()
                    _skip_g_step = True
                    global_step += 1
                    break

                d_grad_norm = clip_gradients(D, args.clip_norm)
                if use_amp_flag:
                    scaler.step(d_optim)
                    scaler.update()
                else:
                    d_optim.step()

                # 更新 EMA 统计 (仅在最后一次 D 迭代)
                if d_iter == d_updates - 1:
                    d_real_ema = 0.99 * d_real_ema + 0.01 * d_real.mean().detach()
                    d_fake_ema = 0.99 * d_fake_ema + 0.01 * d_fake.mean().detach()
                    real_imgs.requires_grad_(False)
                    # ADA 增强概率更新 (基于 D 对真实数据表现)
                    if ada is not None:
                        ada.update_p(d_real.mean().detach().item())
                    # 记录 Aux 分类准确率作为验证指标
                    _acc_real = compute_aux_accuracy(aux_real.detach(), labels)
                    _acc_fake = compute_aux_accuracy(aux_fake.detach(), labels)
                    epoch_aux_acc_real += _acc_real
                    epoch_aux_acc_fake += _acc_fake

                    _perfect_real = compute_perfect_accuracy(aux_real.detach(), labels)
                    _perfect_fake = compute_perfect_accuracy(aux_fake.detach(), labels)
                    epoch_perfect_acc_real += _perfect_real
                    epoch_perfect_acc_fake += _perfect_fake

                    # 捕获 D 训练步的 d_real/d_fake (供日志; 防止被 G 步覆盖)
                    _d_real_dstep = d_real.mean().detach()
                    _d_fake_dstep = d_fake.mean().detach()

            # ═════════════════════════════════════════
            #  训练生成器 G
            # ═════════════════════════════════════════
            if _skip_g_step:
                g_optim.zero_grad()
                continue

            z = torch.randn(B, LATENT_DIM, device=DEVICE)
            g_optim.zero_grad()

            # 同标签双噪声多样性: 半 batch 降本, 给噪声路径直接梯度压力
            _div_half = max(1, B // 2)
            z2 = torch.randn(_div_half, LATENT_DIM, device=DEVICE)

            with autocast():
                fake_imgs = G(z, labels)
                fake_imgs2 = G(z2, labels[:_div_half])
                d_fake, aux_fake, fake_features = D(fake_imgs, labels, return_features=True)

                # 双向对比度匹配: 每图 std 落入真实分布带 [q10-margin, q90+margin],
                # 过高 (过噪) 过低 (灰糊) 都惩罚。
                fake_std = fake_imgs.view(B, -1).std(dim=1)               # (B,)
                with torch.no_grad():
                    real_std = real_imgs.detach().view(B, -1).std(dim=1)  # (B,)
                    _ctr_lo = _batch_quantile(real_std, _CONTRAST_LO_Q) - _CONTRAST_MARGIN
                    _ctr_hi = _batch_quantile(real_std, _CONTRAST_HI_Q) + _CONTRAST_MARGIN
                contrast_loss = (F.relu(_ctr_lo - fake_std) +
                                 F.relu(fake_std - _ctr_hi)).mean()

                cur_edge_loss = edge_loss(fake_imgs, real_imgs)
                cur_realism_loss = realism_loss(fake_imgs, real_imgs)

                # 反粘连 gap 墨迹 hinge (预热后且 --lambda-gap>0 才启用)
                if epoch > G_WARMUP_EPOCHS and args.lambda_gap > 0:
                    cur_gap_loss = gap_ink_loss(fake_imgs)
                else:
                    cur_gap_loss = torch.tensor(0.0, device=DEVICE)
                gap_ink_v = gap_ink_ratio(fake_imgs)

                # 动态 Edge Loss 权重 — 防止边缘监督衰减过快
                edge_lambda = LAMBDA_EDGE
                if hasattr(args, 'edge_loss_min') and cur_edge_loss.item() < args.edge_loss_min:
                    edge_lambda = LAMBDA_EDGE * 2.0

                # ─── OCR 监督 (课程学习) ─────────────────
                # Phase A: 仅 CaptchaResNet, CE 全幅 + FM 不加权 → 冷启动可读性
                # Phase B: 字符级 96 格权重表, CE 固定分母 → 防欺骗
                cur_ocr_lambda = get_ocr_lambda(epoch, batch_idx, fake_imgs, labels)
                if cur_ocr_lambda > 0 and ocr_model is not None:
                    ocr_logits, ocr_feat_f = ocr_model(fake_imgs, return_features=True)
                    _, ocr_feat_r = ocr_model(real_imgs, return_features=True)

                    # Phase C 混淆软目标 CE (t=0 ≡ 硬 CE)
                    if ocr_pc and ocr_pc_t > 0.0:
                        ce_char = soft_ce(ocr_logits, labels,
                                          _SOFT_DIST, _SOFT_EPS, ocr_pc_t)
                    else:
                        ce_char = F.cross_entropy(
                            ocr_logits.view(-1, NUM_CLASSES),
                            labels.view(-1),
                            reduction='none').view(B, CAPTCHA_LENGTH)   # (B,4)

                    # 权重计算 (按阶段选择策略)
                    if ocr_t < 1.0:
                        # Phase A / Ramp: 快速路径 C-only (第三方可能未加载)
                        w_fast, _ = adaptive_ocr.compute_char_weights(
                            fake_imgs.detach(), labels,
                            ocr_logits=ocr_logits.detach(), max_samples=0,
                            phase_c=ocr_pc)
                    if ocr_t > 0.0:
                        # Ramp / Phase B: 分层三方评估 → 96 格主表
                        # (每步快速路径; 每 10/50/100 步抽样/全量三方)
                        if batch_idx % 100 == 0:
                            _n_full = None          # 全量三方
                        elif batch_idx % 50 == 0:
                            _n_full = 20
                        elif batch_idx % 10 == 0:
                            _n_full = 10
                        else:
                            _n_full = 0             # 仅 C 快速路径
                        w_full, _ = adaptive_ocr.compute_char_weights(
                            fake_imgs.detach(), labels,
                            ocr_logits=ocr_logits.detach(), max_samples=_n_full,
                            phase_c=ocr_pc)

                    # CE: Phase A 全幅 ↔ Phase B 固定分母
                    if ocr_t <= 0.0:
                        char_w = w_fast
                        ocr_ce = (ce_char * w_fast).sum() / w_fast.sum().clamp(min=1e-8)
                    elif ocr_t >= 1.0:
                        char_w = w_full
                        ocr_ce = (ce_char * w_full).sum() / float(B * CAPTCHA_LENGTH)
                    else:
                        # Ramp: 权重与归一化方式同步线性混合
                        char_w = (1.0 - ocr_t) * w_fast + ocr_t * w_full
                        _ce_a = (ce_char * w_fast).sum() / w_fast.sum().clamp(min=1e-8)
                        _ce_b = (ce_char * w_full).sum() / float(B * CAPTCHA_LENGTH)
                        ocr_ce = (1.0 - ocr_t) * _ce_a + ocr_t * _ce_b

                    # 特征匹配全程不加权 (批级分布匹配, 不诱导欺骗)
                    ocr_feat_m = ocr_net.ocr_feature_loss(ocr_feat_f, ocr_feat_r)

                    ocr_loss_val = (OCR_CE_WEIGHT * ocr_ce +
                                    (1.0 - OCR_CE_WEIGHT) * ocr_feat_m)

                    ocr_w_mean_v = float(char_w.mean())
                    ocr_valid_ratio = float((char_w > OCR_VALID_THR).float().mean())
                    ocr_acc_item = (ocr_logits.detach().argmax(-1) == labels
                                    ).all(dim=1).float().mean()
                else:
                    ocr_ce = torch.tensor(0.0, device=DEVICE)
                    ocr_feat_m = torch.tensor(0.0, device=DEVICE)
                    ocr_loss_val = torch.tensor(0.0, device=DEVICE)
                    ocr_acc_item = torch.tensor(0.0, device=DEVICE)
                    ocr_w_mean_v = 0.0
                    ocr_valid_ratio = 0.0

                # 第三方 OCR 准确率 (复用 check_and_update 的计算, 零额外开销)
                # 未测量记 -1.0 (Phase A 不加载第三方 OCR), 区分"未测"与"真0"
                dddd_acc_v = ppll_acc_v = -1.0
                if adaptive_ocr is not None:
                    _ah = getattr(adaptive_ocr, "accuracy_history", {})
                    if _ah.get("ddddocr"):
                        dddd_acc_v = float(_ah["ddddocr"][-1])
                    if _ah.get("ppllocr"):
                        ppll_acc_v = float(_ah["ppllocr"][-1])

                if in_g_warmup:
                    # ═══ G 预热: Aux + FM + Contrast + Diversity + Edge + Realism ═══
                    aux_loss = aux_criterion(
                        aux_fake.view(-1, NUM_CLASSES), labels.view(-1))
                    div_loss = diversity_loss(fake_imgs[:_div_half], fake_imgs2)
                    if cached_real_features is not None:
                        fm_loss = feature_matching_loss(cached_real_features, fake_features)
                    else:
                        fm_loss = torch.tensor(0.0, device=DEVICE)
                    g_loss_total = (LAMBDA_AUX_G * aux_loss +
                                    LAMBDA_FM * fm_loss +
                                    LAMBDA_CONTRAST * contrast_loss +
                                    LAMBDA_DIVERSITY * div_loss +
                                    edge_lambda * cur_edge_loss +
                                    LAMBDA_REALISM * cur_realism_loss +
                                    args.lambda_gap * cur_gap_loss +
                                    cur_ocr_lambda * ocr_loss_val)
                    cur_adv_lambda = 0.0
                    cur_fm_lambda = LAMBDA_FM
                else:
                    # ═══ 渐进对抗训练: 对抗 + Aux + FM 三足鼎立 + 视觉质量约束 ═══
                    cur_adv_lambda = get_adv_lambda(epoch, G_WARMUP_EPOCHS)
                    g_loss_adv = -d_fake.mean() * cur_adv_lambda

                    aux_loss = aux_criterion(
                        aux_fake.view(-1, NUM_CLASSES), labels.view(-1))

                    cur_fm_lambda = LAMBDA_FM
                    if cached_real_features is not None:
                        fm_loss = feature_matching_loss(cached_real_features, fake_features)
                    else:
                        fm_loss = torch.tensor(0.0, device=DEVICE)

                    div_loss = diversity_loss(fake_imgs[:_div_half], fake_imgs2)

                    g_loss_total = (g_loss_adv +
                                    LAMBDA_AUX_G * aux_loss +
                                    cur_fm_lambda * fm_loss +
                                    LAMBDA_CONTRAST * contrast_loss +
                                    LAMBDA_DIVERSITY * div_loss +
                                    edge_lambda * cur_edge_loss +
                                    LAMBDA_REALISM * cur_realism_loss +
                                    args.lambda_gap * cur_gap_loss +
                                    cur_ocr_lambda * ocr_loss_val)

            # detach 替代 .item(), 避免 per-step 设备同步 (日志间隔才 sync)
            g_item = g_loss_total.detach()
            fm_item = fm_loss.detach()
            contrast_item = contrast_loss.detach()
            div_item = div_loss.detach()
            edge_item = cur_edge_loss.detach()
            realism_item = cur_realism_loss.detach()
            ocr_item = ocr_loss_val.detach()
            ocr_ce_item = ocr_ce.detach()
            ocr_feat_item = ocr_feat_m.detach()

            if use_amp_flag:
                scaler.scale(g_loss_total).backward()
                scaler_unscale_(scaler, g_optim)
            else:
                g_loss_total.backward()

            # G 梯度 NaN 检测
            _g_grad_ok = True
            for _name, _p in G.named_parameters():
                if _p.grad is not None and (not torch.isfinite(_p.grad).all()):
                    _g_grad_ok = False
                    break
            if not _g_grad_ok:
                _consecutive_nan += 1
                if _consecutive_nan > 500:
                    print(f"  [FATAL] 连续 NaN 超过 500 次 (step {global_step}), 请降低学习率或增大 init_scale")
                    raise RuntimeError("训练崩溃: 连续 NaN 过多, 模型权重可能已损坏")
                print(f"  [WARN] G 梯度 NaN at step {global_step} (连续{_consecutive_nan}次), 跳过本步 G 更新")
                g_optim.zero_grad()
                if use_amp_flag:
                    scaler.update()
                global_step += 1
                continue

            g_grad_norm = clip_gradients(G, args.clip_norm)
            if use_amp_flag:
                scaler.step(g_optim)
                scaler.update()
            else:
                g_optim.step()

            # EMA 更新 (每步都更新, 预热期也更新)
            ema_g.update(G)

            global_step += 1

            # ─── 记录 ────────────────────────────────
            _consecutive_nan = 0  # 正常步, 重置连续 NaN 计数
            d_item = d_loss_total.detach()
            aux_item = aux_loss.detach()

            # NaN 兜底检测
            if torch.isnan(d_loss_total) or torch.isnan(g_loss_total):
                _consecutive_nan += 1
                print(f"  [WARN] Loss NaN at step {global_step} (连续{_consecutive_nan}次), 跳过记录和 EMA")
                global_step += 1
                continue

            epoch_g_loss += g_item
            epoch_d_loss += d_item
            epoch_aux_loss += aux_item
            epoch_fm_loss += fm_item
            epoch_contrast_loss += contrast_item
            epoch_div_loss += div_item
            epoch_edge_loss += edge_item
            epoch_realism_loss += realism_item
            epoch_ocr_loss += ocr_item
            epoch_ocr_ce += ocr_ce_item
            epoch_ocr_feat += ocr_feat_item
            epoch_ocr_lambda += cur_ocr_lambda
            epoch_ocr_w_mean += ocr_w_mean_v
            epoch_ocr_valid += ocr_valid_ratio
            epoch_ocr_acc += ocr_acc_item.detach()
            if dddd_acc_v >= 0:                           # 仅统计实测
                epoch_dddd_acc += dddd_acc_v
                epoch_dddd_n += 1
            if ppll_acc_v >= 0:
                epoch_ppll_acc += ppll_acc_v
                epoch_ppll_n += 1
            epoch_gap_ink += gap_ink_v
            epoch_batches += 1

            # ─── 控制台日志 ───────────────────────────
            if global_step % args.log_interval == 0:
                elapsed = time.time() - t_start
                ips = global_step / elapsed if elapsed > 0 else 0
                cur_lr_g = g_optim.param_groups[0]["lr"]
                cur_lr_d = d_optim.param_groups[0]["lr"]
                r1_item = r1_penalty_val.item() if isinstance(r1_penalty_val, torch.Tensor) else 0.0
                r2_item = r2_penalty_val.item() if isinstance(r2_penalty_val, torch.Tensor) else 0.0
                cur_adv_lambda = get_adv_lambda(epoch, G_WARMUP_EPOCHS) if not in_g_warmup else 0.0
                _d_val = d_item.item()
                _g_val = g_item.item()
                _aux_val = aux_item.item()
                _fm_val = fm_item.item()
                _ctr_val = contrast_item.item()
                _div_val = div_item.item()
                _edge_val = edge_item.item()
                _real_val = realism_item.item()
                _dR_val = _d_real_dstep.item()
                _dF_val = _d_fake_dstep.item()
                # 日志用 CaptchaResNet 整串准确率
                if ocr_model is not None:
                    _ocr_acc = ocr_net.ocr_accuracy(ocr_model, fake_imgs, labels)
                    _ocr_acc_v = _ocr_acc.item()
                else:
                    _ocr_acc_v = 0.0
                _valid_ratio = ocr_valid_ratio
                _ocr_w_mean = ocr_w_mean_v
                phase = "WARM" if in_g_warmup else ("ADV_WARM" if in_adv_warmup else "JOINT")
                print(
                    f"  [{phase}] Epoch {epoch:3d}/{args.epochs} "
                    f"[{batch_idx:4d}/{len(loader):4d}] "
                    f"Step {global_step:6d} | "
                    f"D: {_d_val:.3f} "
                    f"G: {_g_val:.3f} "
                    f"Aux: {_aux_val:.3f} "
                    f"OCR[{ocr_ph}]: {_ocr_acc_v:.3f}(w={_ocr_w_mean:.3f},效{_valid_ratio:.0%}) "
                    f"FM: {_fm_val:.3f}(λ={LAMBDA_FM:.1f}) "
                    f"Div: {_div_val:.4f} "
                    f"Edge: {_edge_val:.4f} "
                    f"Real: {_real_val:.4f} "
                    f"Adv: {cur_adv_lambda:.2f} "
                    f"Ctr: {_ctr_val:.3f} "
                    f"R1: {r1_item:.4f}(λ={cur_r1_lambda:.2f}) "
                    f"R2: {r2_item:.4f}(λ={cur_r2_lambda:.2f}) "
                    f"|dG|={d_grad_norm:.1f} |gG|={g_grad_norm:.1f} | "
                    f"dR={_dR_val:+.2f} "
                    f"dF={_dF_val:+.2f} "
                    f"Δ={_dR_val - _dF_val:+.2f} | "
                    f"D↑={d_updates} | "
                    f"LR: G={cur_lr_g:.2e} D={cur_lr_d:.2e} | "
                    f"{ips:.1f}步/秒"
                    f"{' ADA:' + str(ada.p)[:5] if ada is not None else ''}"
                )

            # ─── CSV 文件日志 ─────────────────────────
            if global_step % _FILE_LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                ips = global_step / elapsed if elapsed > 0 else 0
                cur_lr_g_v = g_optim.param_groups[0]["lr"]
                cur_lr_d_v = d_optim.param_groups[0]["lr"]
                r1_v = r1_penalty_val.item() if isinstance(r1_penalty_val, torch.Tensor) else 0.0
                r2_v = r2_penalty_val.item() if isinstance(r2_penalty_val, torch.Tensor) else 0.0
                # FM 权重恒为 LAMBDA_FM, 直接记录真实值
                fm_l_v = float(LAMBDA_FM)
                _step_acc_r = _acc_real.item() if isinstance(_acc_real, torch.Tensor) else _acc_real
                _step_acc_f = _acc_fake.item() if isinstance(_acc_fake, torch.Tensor) else _acc_fake
                _dR = _d_real_dstep.item()
                _dF = _d_fake_dstep.item()
                _dRE = d_real_ema.item()
                _dFE = d_fake_ema.item()
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _write_log_line(
                    log_file, ts, epoch, global_step, phase,
                    d_item.item(), g_item.item(), aux_item.item(),
                    fm_item.item(), fm_l_v, contrast_item.item(),
                    div_item.item(),
                    edge_item.item(),
                    realism_item.item(),
                    r1_v, r2_v,
                    _dR, _dF,
                    _dRE, _dFE,
                    d_grad_norm, g_grad_norm,
                    cur_lr_g_v, cur_lr_d_v,
                    _step_acc_r, _step_acc_f,
                    _perfect_real.item(), _perfect_fake.item(),
                    ips, B,
                    ocr_loss=ocr_item.item(), ocr_ce=ocr_ce_item.item(),
                    ocr_feat=ocr_feat_item.item(), ocr_lambda=float(cur_ocr_lambda),
                    ocr_w_mean=float(ocr_w_mean_v), ocr_valid_ratio=float(ocr_valid_ratio),
                    ocr_acc=float(ocr_acc_item),
                    dddd_acc=dddd_acc_v, ppll_acc=ppll_acc_v,
                    gap_ink=gap_ink_v,
                    ocr_phase=ocr_ph, ocr_ramp=ocr_ramp_v,
                )

        # ─── 每个 epoch 结束 ────────────────────────────
        epoch_batches_safe = max(epoch_batches, 1)
        avg_g = (epoch_g_loss / epoch_batches_safe).item()
        avg_d = (epoch_d_loss / epoch_batches_safe).item()
        avg_aux = (epoch_aux_loss / epoch_batches_safe).item()
        avg_fm = (epoch_fm_loss / epoch_batches_safe).item()
        avg_ctr = (epoch_contrast_loss / epoch_batches_safe).item()
        avg_div = (epoch_div_loss / epoch_batches_safe).item()
        avg_edge = (epoch_edge_loss / epoch_batches_safe).item()
        avg_real = (epoch_realism_loss / epoch_batches_safe).item()
        avg_ocr = (epoch_ocr_loss / epoch_batches_safe).item()
        avg_ocr_ce = (epoch_ocr_ce / epoch_batches_safe).item()
        avg_ocr_feat = (epoch_ocr_feat / epoch_batches_safe).item()
        avg_ocr_lambda = epoch_ocr_lambda / epoch_batches_safe
        avg_ocr_w = epoch_ocr_w_mean / epoch_batches_safe
        avg_ocr_valid = epoch_ocr_valid / epoch_batches_safe
        avg_ocr_acc = (epoch_ocr_acc / epoch_batches_safe).item()
        avg_dddd_acc = (epoch_dddd_acc / epoch_dddd_n) if epoch_dddd_n > 0 else -1.0
        avg_ppll_acc = (epoch_ppll_acc / epoch_ppll_n) if epoch_ppll_n > 0 else -1.0
        avg_gap_ink = epoch_gap_ink / epoch_batches_safe
        avg_acc_real = epoch_aux_acc_real / epoch_batches_safe
        avg_acc_fake = epoch_aux_acc_fake / epoch_batches_safe
        avg_perfect_real = epoch_perfect_acc_real / epoch_batches_safe
        avg_perfect_fake = epoch_perfect_acc_fake / epoch_batches_safe

        # 学习率调度
        g_scheduler.step()
        d_scheduler.step()

        # 样例生成 (EMA 模型 + 训练 G 对比)
        if epoch % args.sample_interval == 0 or epoch == 1:
            save_sample(ema_g.ema_model, fixed_z, fixed_labels,
                        epoch, global_step, output_dir, tag="ema")
            save_sample(G, fixed_z2, fixed_labels2,
                        epoch, global_step, output_dir, tag="train")

        # 模型保存 (含 EMA 与 scheduler 状态)
        if epoch % args.save_interval == 0 or epoch == 1:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "generator_state_dict": G.state_dict(),
                "discriminator_state_dict": D.state_dict(),
                "g_optimizer_state_dict": g_optim.state_dict(),
                "d_optimizer_state_dict": d_optim.state_dict(),
                "ema_state_dict": ema_g.state_dict(),
                "g_scheduler_state_dict": g_scheduler.state_dict(),
                "d_scheduler_state_dict": d_scheduler.state_dict(),
            }, ckpt_path)
            print(f"  [SAVE] {ckpt_path.name}")

        epoch_time = time.time() - epoch_start
        cur_lr = g_optim.param_groups[0]["lr"]
        cur_adv = get_adv_lambda(epoch, G_WARMUP_EPOCHS) if not in_g_warmup else 0.0
        phase = "WARM" if in_g_warmup else ("ADV_WARM" if in_adv_warmup else "JOINT")

        ocr_status = ""
        if adaptive_ocr is not None:
            ocr_status = f" | OCR权重={adaptive_ocr.current_weight:.4f}"

        print(
            f"  [{phase}] —— Epoch {epoch:3d} 完成 "
            f"| D={avg_d:.3f} G={avg_g:.3f} Aux={avg_aux:.3f} "
            f"FM={avg_fm:.3f}(λ={float(LAMBDA_FM):.2f}) Div={avg_div:.4f} "
            f"Edge={avg_edge:.4f} Real={avg_real:.4f} "
            f"OCR[{ocr_ph}]={avg_ocr:.4f}(CE={avg_ocr_ce:.3f},w={avg_ocr_w:.3f},λ={avg_ocr_lambda:.2f}) "
            f"Adv={cur_adv:.2f} Ctr={avg_ctr:.3f} "
            f"AccR={avg_acc_real:.3f} AccF={avg_acc_fake:.3f} "
            f"PerfR={avg_perfect_real:.3f} PerfF={avg_perfect_fake:.3f} "
            f"dR_ema={d_real_ema.item():+.2f} dF_ema={d_fake_ema.item():+.2f} "
            f"LR={cur_lr:.2e}{ocr_status} | 耗时 {epoch_time:.0f}s ——\n"
        )

        # CSV: epoch 汇总行
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _write_log_line(
            log_file, ts, epoch, global_step, f"{phase}_EPOCH",
            avg_d, avg_g, avg_aux,
            avg_fm, float(LAMBDA_FM), avg_ctr, avg_div,
            avg_edge,
            avg_real,
            0.0, 0.0,  # epoch 汇总行不记录 R1/R2
            d_real_ema.item(), d_fake_ema.item(),
            d_real_ema.item(), d_fake_ema.item(),
            0.0, 0.0,
            cur_lr, cur_lr,
            avg_acc_real, avg_acc_fake,
            avg_perfect_real, avg_perfect_fake,
            0.0, 0,
            ocr_loss=avg_ocr, ocr_ce=avg_ocr_ce,
            ocr_feat=avg_ocr_feat, ocr_lambda=avg_ocr_lambda,
            ocr_w_mean=avg_ocr_w, ocr_valid_ratio=avg_ocr_valid,
            ocr_acc=avg_ocr_acc,
            dddd_acc=avg_dddd_acc, ppll_acc=avg_ppll_acc,
            gap_ink=avg_gap_ink,
            ocr_phase=ocr_ph, ocr_ramp=ocr_ramp_v,
        )

        # 清理缓存 (降低频率, 避免频繁内存碎片整理)
        if epoch % 10 == 0:
            empty_cache()

    # ─── 训练完成 ──────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  训练完成！总耗时 {total_time/60:.1f} 分钟")
    print(f"  模型保存在: {output_dir.resolve()}")
    print(f"  日志文件:    {(output_dir / 'training_log.csv').resolve()}")
    print(f"{'='*60}")

    log_file.close()

    # 保存最终模型
    final_ckpt = output_dir / f"checkpoint_epoch_{args.epochs:03d}.pt"
    torch.save({
        "epoch": args.epochs,
        "global_step": global_step,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "g_optimizer_state_dict": g_optim.state_dict(),
        "d_optimizer_state_dict": d_optim.state_dict(),
        "ema_state_dict": ema_g.state_dict(),
        "g_scheduler_state_dict": g_scheduler.state_dict(),
        "d_scheduler_state_dict": d_scheduler.state_dict(),
    }, final_ckpt)
    print(f"  [SAVE] {final_ckpt.name}")
    save_sample(ema_g.ema_model, fixed_z, fixed_labels, args.epochs, global_step, output_dir)


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="验证码 AC-GAN 训练 (增强版)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help=f"训练数据目录 (默认: {DATA_DIR})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"模型和采样输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"训练轮数 (默认: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch 大小 (默认: {BATCH_SIZE})")
    parser.add_argument("--lr-g", type=float, default=LEARNING_RATE_G,
                        help=f"Generator 学习率 (默认: {LEARNING_RATE_G})")
    parser.add_argument("--lr-d", type=float, default=LEARNING_RATE_D,
                        help=f"Discriminator 学习率 (默认: {LEARNING_RATE_D})")
    parser.add_argument("--beta1", type=float, default=BETA1,
                        help=f"Adam beta1 (默认: {BETA1})")
    parser.add_argument("--beta2", type=float, default=BETA2,
                        help=f"Adam beta2 (默认: {BETA2})")
    parser.add_argument("--d-updates", type=int, default=D_UPDATES,
                        help="D 基础更新次数 (被自适应策略动态覆盖)")
    parser.add_argument("--clip-norm", type=float, default=CLIP_NORM,
                        help=f"梯度裁剪上限 (默认: {CLIP_NORM})")
    parser.add_argument("--d-pretrain", type=int, default=D_PRETRAIN_EPOCHS,
                        help=f"D 预训练 epoch 数 (默认: {D_PRETRAIN_EPOCHS})")
    parser.add_argument("--resume", type=str, default=None,
                        help="检查点路径，从断点续训")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL,
                        help=f"日志打印间隔 (步数, 默认: {LOG_INTERVAL})")
    parser.add_argument("--sample-interval", type=int, default=1,
                        help="采样间隔 (epoch, 默认: 1)")
    parser.add_argument("--save-interval", type=int, default=MODEL_SAVE_INTERVAL,
                        help=f"模型保存间隔 (epoch, 默认: {MODEL_SAVE_INTERVAL})")
    parser.add_argument("--use-cosine-lr", action="store_true", default=USE_COSINE_LR,
                        help="使用 Cosine Annealing 学习率调度器")
    parser.add_argument("--lr-min", type=float, default=LR_MIN,
                        help=f"Cosine 调度器最小学习率 (默认: {LR_MIN})")
    parser.add_argument("--edge-loss-min", type=float, default=EDGE_LOSS_MIN,
                        help=f"Edge Loss 下限, 低于此值时权重翻倍 (默认: {EDGE_LOSS_MIN})")
    parser.add_argument("--curriculum-epochs", type=int, default=CURRICULUM_EPOCHS,
                        help=f"OCR课程学习 Phase A (仅CaptchaResNet强引导) 轮数 (默认: {CURRICULUM_EPOCHS})")
    parser.add_argument("--curriculum-ramp", type=int, default=CURRICULUM_RAMP,
                        help=f"OCR课程学习 A→B 切换期轮数 (默认: {CURRICULUM_RAMP}, 建议2-5)")
    parser.add_argument("--no-curriculum", action="store_true", default=not USE_CURRICULUM,
                        help="禁用OCR课程学习, 全程使用三OCR方案")
    parser.add_argument("--phase-c-epoch", type=int, default=PHASE_C_EPOCHS,
                        help=f"Phase C 混淆软目标起始轮 (默认: {PHASE_C_EPOCHS}, 0=禁用)")
    parser.add_argument("--phase-c-ramp", type=int, default=PHASE_C_RAMP,
                        help=f"Phase C ε 线性爬升轮数 (默认: {PHASE_C_RAMP})")
    parser.add_argument("--lambda-gap", type=float, default=LAMBDA_GAP,
                        help=f"反粘连 gap hinge 损失权重 (0=关闭, 默认 {LAMBDA_GAP})")
    return parser.parse_args()


def main():
    args = parse_args()

    # 命令行路径覆盖 config 默认值
    if args.data_dir:
        import data_loader
        data_loader.DATA_DIR = Path(args.data_dir)
    if args.output_dir:
        import utils as utils_mod
        utils_mod.OUTPUT_DIR = Path(args.output_dir)

    setup_matmul_precision()
    print_npu_env_hints()

    train(args)


if __name__ == "__main__":
    main()
