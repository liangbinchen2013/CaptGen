"""
工具函数: 样例可视化、模型加载
"""
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
    """
    生成并保存样例图片。

    tag 参数:
      - "ema":   使用 EMA 模型 (eval 模式, BN 用 running stats)
      - "train": 使用训练中的 G (train 模式, BN 用 batch stats),
                 反映训练时的真实生成质量
    """
    was_training = generator.training
    if tag != "train":
        generator.eval()

    with torch.no_grad():
        fake = generator(fixed_z, fixed_labels)

    # 仅临时切换模式: 采样结束后恢复训练状态
    if was_training and tag != "train":
        generator.train()

    # 反归一化 [-1,1] → [0,255]
    fake = (fake.cpu() * 0.5 + 0.5).clamp(0, 1)

    # 按行排列
    nrow = min(8, fixed_z.shape[0])
    grid = vutils.make_grid(fake, nrow=nrow, padding=2, pad_value=1)
    # make_grid 返回 3 通道, 取第一通道即可
    grid = grid.numpy()[0] * 255
    grid = grid.astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"epoch_{epoch:03d}_step_{step:06d}_{tag}.png"
    Image.fromarray(grid).save(path)

    # 保存对应的标签
    labels_str = []
    for i in range(fixed_labels.shape[0]):
        label = ""
        for j in range(CAPTCHA_LENGTH):
            label += CHARS[fixed_labels[i, j].item()]
        labels_str.append(label)

    with open(output_dir / f"sample_labels_{tag}.txt", "a", encoding="utf-8") as f:
        f.write(f"epoch={epoch:03d} step={step:06d}: {','.join(labels_str)}\n")

    print(f"  [SAMPLE-{tag.upper()}] {path.name} | labels: {labels_str[:4]}...")
    return path


def load_model(path: Path, generator, discriminator, g_optim=None, d_optim=None, device=None):
    """加载模型检查点 (discriminator/g_optim/d_optim 可为 None)

    Generator 优先加载 EMA 权重 (更稳定), 否则加载原始权重。
    """
    if device is None:
        device = DEVICE
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if generator is not None:
        if "ema_state_dict" in checkpoint:
            generator.load_state_dict(checkpoint["ema_state_dict"])
            print(f"[OK] 已加载 EMA Generator 权重")
        else:
            generator.load_state_dict(checkpoint["generator_state_dict"])
            print(f"[OK] 已加载 Generator 权重")

    if discriminator is not None:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    if g_optim is not None and "g_optimizer_state_dict" in checkpoint:
        g_optim.load_state_dict(checkpoint["g_optimizer_state_dict"])
    if d_optim is not None and "d_optimizer_state_dict" in checkpoint:
        d_optim.load_state_dict(checkpoint["d_optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    print(f"[OK] 已加载 {path.name}, epoch={epoch}")
    return epoch
