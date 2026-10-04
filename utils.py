"""
工具函数：可视化、模型保存、样例生成
"""
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image

from config import *
from models import Generator
from device_utils import DEVICE


def save_sample(generator: Generator, fixed_z: torch.Tensor,
                fixed_labels: torch.Tensor, epoch: int, step: int,
                output_dir: Path, tag: str = "ema"):
    """生成并保存样例图片

    v5: 新增 tag 参数, 支持区分 EMA 和训练 G 的采样输出。
        - tag="ema": 使用 eval 模式 (BN running stats, v5 已修复同步)
        - tag="train": 使用 train 模式 (BN batch stats), 反映训练时的真实生成质量

    注意: 对于 tag="train", generator 必须处于 train 模式,
    调用者负责在采样前后切换 generator.eval()/train()。
    """
    is_training = generator.training
    if tag == "train":
        # v5: 训练模式采样 — 使用 batch stats, 反映 G 的真实生成能力
        # 不需要切换到 eval, 保持 train 模式让 BN 使用当前 batch 统计量
        pass
    else:
        generator.eval()

    with torch.no_grad():
        fake = generator(fixed_z, fixed_labels)

    # 恢复原始模式
    if tag == "train" and is_training:
        pass  # 保持 train 模式
    elif not is_training:
        generator.train()

    # 反归一化 [−1,1] → [0,255]
    fake = (fake.cpu() * 0.5 + 0.5).clamp(0, 1)

    # 按行排列
    nrow = min(8, fixed_z.shape[0])
    grid = vutils.make_grid(fake, nrow=nrow, padding=2, pad_value=1)
    # make_grid always returns 3-channel RGB; take the first channel
    grid = grid.numpy()[0] * 255
    grid = grid.astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    # v5: 文件名包含 tag 以区分 EMA vs 训练 G
    path = output_dir / f"epoch_{epoch:03d}_step_{step:06d}_{tag}.png"
    Image.fromarray(grid, mode="L").save(path)

    # 同时保存带标签的版本
    labels_str = []
    for i in range(fixed_labels.shape[0]):
        label = ""
        for j in range(CAPTCHA_LENGTH):
            label += CHARS[fixed_labels[i, j].item()]
        labels_str.append(label)

    # 保存 txt 标注
    with open(output_dir / f"sample_labels_{tag}.txt", "a", encoding="utf-8") as f:
        f.write(f"epoch={epoch:03d} step={step:06d}: {','.join(labels_str)}\n")

    print(f"  [SAMPLE-{tag.upper()}] {path.name} | labels: {labels_str[:4]}...")
    return path


def save_model(generator, discriminator, g_optim, d_optim, epoch, output_dir):
    """保存模型检查点"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save({
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "g_optimizer_state_dict": g_optim.state_dict(),
        "d_optimizer_state_dict": d_optim.state_dict(),
    }, path)
    print(f"  [SAVE] {path.name}")
    return path


def load_model(path: Path, generator, discriminator, g_optim=None, d_optim=None, device=None):
    """加载模型检查点 (discriminator/g_optim/d_optim 可为 None)"""
    if device is None:
        device = DEVICE
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # Generator: 优先加载 EMA 权重 (更稳定), 否则加载原始权重
    if generator is not None:
        if "ema_state_dict" in checkpoint:
            generator.load_state_dict(checkpoint["ema_state_dict"])
            print(f"[OK] 已加载 EMA Generator 权重")
        else:
            generator.load_state_dict(checkpoint["generator_state_dict"])
            print(f"[OK] 已加载 Generator 权重")

    # Discriminator: 仅在需要时加载
    if discriminator is not None:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    if g_optim is not None and "g_optimizer_state_dict" in checkpoint:
        g_optim.load_state_dict(checkpoint["g_optimizer_state_dict"])
    if d_optim is not None and "d_optimizer_state_dict" in checkpoint:
        d_optim.load_state_dict(checkpoint["d_optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    print(f"[OK] 已加载 {path.name}, epoch={epoch}")
    return epoch
