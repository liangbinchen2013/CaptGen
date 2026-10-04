"""
生成多样性验证 — 检验噪声路径是否生效 (同标签不同 z 应产生不同图像)

背景:
  若生成器学会忽略噪声 z, 会退化为"标签→图像"的确定性函数 (严重模式坍塌),
  同标签不同 z 的像素差趋近于 0。本项目通过推理时噪声注入 +
  同标签双噪声对比损失强制 z 影响输出。

指标:
  1. 同标签多 z 采样 → 像素级 std (0-255), 目标 > 1.0
  2. 亮度分布 (暗底/亮底/中间态) 与真实数据对比
  3. 字符可读性 (D Aux 识别) 不下降

用法:
  python verify_diversity.py                                    # 最新 checkpoint
  python verify_diversity.py --checkpoint ./output_v28/checkpoint_epoch_400.pt
"""
import argparse
from pathlib import Path

import torch
import numpy as np

from config import *
from models import Generator, Discriminator
from utils import load_model
from device_utils import DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="生成多样性验证")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型检查点路径 (默认自动选择最新)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"checkpoint 搜索目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--labels", type=int, default=40,
                        help="测试标签数量")
    parser.add_argument("--repeats", type=int, default=5,
                        help="每个标签重复采样的 z 数量")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="多样性通过阈值 (同标签像素std, 0-255)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="真实数据目录 (用于亮度分布对比)")
    return parser.parse_args()


def find_best_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(Path(output_dir).glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"未找到 checkpoint 文件: {output_dir}")
    return checkpoints[-1]


def load_generator(ckpt_path: Path):
    G = Generator().to(DEVICE).eval()
    load_model(ckpt_path, G, None, device=DEVICE)
    return G


def measure_diversity(G, num_labels: int, repeats: int) -> float:
    """同标签不同 z 采样的像素级 std (0-255). 返回值越大多样性越好"""
    all_stds = []
    torch.manual_seed(1234)
    labels_pool = torch.randint(0, NUM_CLASSES, (num_labels, CAPTCHA_LENGTH),
                                device=DEVICE)
    with torch.no_grad():
        for i in range(num_labels):
            label = labels_pool[i:i + 1].repeat(repeats, 1)
            z = torch.randn(repeats, LATENT_DIM, device=DEVICE)
            fake = G(z, label).cpu()
            imgs = (fake * 0.5 + 0.5).clamp(0, 1).numpy() * 255  # (R,1,H,W)
            std = imgs.std(axis=0).mean()
            all_stds.append(std)
    return float(np.mean(all_stds))


def brightness_distribution(imgs_np: np.ndarray) -> tuple:
    """imgs_np: (N, H, W) uint8 → (暗底占比, 亮底占比, 中间占比)"""
    means = imgs_np.reshape(len(imgs_np), -1).mean(axis=1)
    dark = (means < 50).mean()
    bright = (means > 200).mean()
    return float(dark), float(bright), float(1 - dark - bright)


def measure_brightness(G, num_samples: int = 200) -> tuple:
    """生成图的亮度分布"""
    torch.manual_seed(5678)
    labels = torch.randint(0, NUM_CLASSES, (num_samples, CAPTCHA_LENGTH),
                           device=DEVICE)
    z = torch.randn(num_samples, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        fake = G(z, labels).cpu()
    imgs = (fake * 0.5 + 0.5).clamp(0, 1).numpy() * 255
    return brightness_distribution(imgs[:, 0])


def real_brightness_distribution(data_dir: Path, num_samples: int = 300) -> tuple:
    """真实数据亮度分布"""
    import random
    from PIL import Image
    random.seed(99)
    files = sorted(Path(data_dir).glob("batch_*/captcha_*.jpg"))
    sample = random.sample(files, min(num_samples, len(files)))
    means = []
    for f in sample:
        a = np.array(Image.open(f).convert("L"), dtype=np.float32)
        means.append(a.mean())
    means = np.array(means)
    dark = (means < 50).mean()
    bright = (means > 200).mean()
    return float(dark), float(bright), float(1 - dark - bright)


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(args.output_dir)
    print(f"Checkpoint: {ckpt_path}")

    G = load_generator(ckpt_path)

    # 1. 多样性
    div = measure_diversity(G, args.labels, args.repeats)
    print(f"\n── 同标签多z多样性 (阈值: {args.threshold}) ──")
    print(f"同标签像素std: {div:.3f} / 255")
    if div > args.threshold:
        print(f"  [PASS] z 路径已恢复, 同标签可生成多样图像")
    else:
        print(f"  [FAIL] 噪声被忽略, 同标签输出趋同 (模式坍塌)")

    # 2. 亮度分布
    g_dark, g_bright, g_mid = measure_brightness(G)
    r_dark, r_bright, r_mid = real_brightness_distribution(args.data_dir)
    print(f"\n── 亮度分布 (暗底/中间/亮底) ──")
    print(f"生成: {g_dark*100:.1f}% / {g_mid*100:.1f}% / {g_bright*100:.1f}%")
    print(f"真实: {r_dark*100:.1f}% / {r_mid*100:.1f}% / {r_bright*100:.1f}%")
    if abs(g_mid - r_mid) > 0.1:
        print(f"  [WARN] 中间亮度缺失: 生成 {g_mid*100:.1f}% vs 真实 {r_mid*100:.1f}%")

    # 3. 字符可读性 (D Aux)
    print(f"\n── 字符可读性 (D Aux) ──")
    D = Discriminator().to(DEVICE).eval()
    load_model(ckpt_path, None, D, device=DEVICE)
    torch.manual_seed(42)
    labels = torch.randint(0, NUM_CLASSES, (128, CAPTCHA_LENGTH), device=DEVICE)
    z = torch.randn(128, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        fake = G(z, labels)
        _, aux, _ = D(fake, labels)
    acc = (aux.argmax(-1) == labels).float().mean().item()
    print(f"Aux 逐字符准确率: {acc*100:.1f}%")
    if acc < 0.7:
        print(f"  [WARN] 可读性下降, 需要检查字符渲染质量")

    print(f"\n{'='*40}")
    overall = "PASS" if div > args.threshold else "FAIL"
    print(f"综合判定: {overall}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
