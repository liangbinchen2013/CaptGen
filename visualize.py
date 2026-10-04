"""
查看数据集样本和生成结果

用法:
  python visualize.py --samples                                          # 查看训练集样本
  python visualize.py --samples --data-dir /path/to/训练数据              # 指定数据路径
  python visualize.py --generated --input-dir ./my_captchas              # 查看生成结果
  python visualize.py --generated --output-dir ./output                  # 保存预览到指定目录
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import *


def show_training_samples(count=16, data_dir=None, output_dir=None):
    """显示训练集中的样本"""
    if data_dir is None:
        data_dir = DATA_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    from data_loader import CaptchaDataset
    dataset = CaptchaDataset(data_dir)

    fig, axes = plt.subplots(4, 4, figsize=(10, 5))
    axes = axes.flatten()

    indices = np.random.choice(len(dataset), count, replace=False)
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img = (img * 0.5 + 0.5).numpy()[0]  # [-1,1] → [0,1]
        label_str = "".join(CHARS[li] for li in label)
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(label_str, fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "training_samples.png"
    plt.savefig(str(out), dpi=150)
    print(f"[OK] 训练样本已保存: {out}")


def show_generated_results(input_dir="generated", output_dir=None, count=16):
    """显示生成的结果"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    gen_dir = Path(input_dir)
    if not gen_dir.exists():
        print(f"[WARN] 未找到 {gen_dir}/ 目录，请先运行 generate.py")
        return

    png_files = sorted(gen_dir.glob("gen_*.png"))[:count]
    if not png_files:
        print(f"[WARN] {gen_dir}/ 下未找到生成图片 (gen_*.png)")
        return

    rows = int(np.ceil(len(png_files) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i, f in enumerate(png_files):
        img = Image.open(f).convert("L")
        label = f.stem.replace("gen_", "")
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(label, fontsize=10)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "generated_preview.png"
    plt.savefig(str(out), dpi=150)
    print(f"[OK] 生成预览已保存: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="store_true", help="显示训练集样本")
    parser.add_argument("--generated", action="store_true", help="显示生成结果")
    parser.add_argument("--data-dir", type=str, default=None,
                        help=f"训练数据路径 (默认: {DATA_DIR})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"预览图保存目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--input-dir", type=str, default="generated",
                        help="生成图片所在目录 (默认: generated/)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.samples:
        show_training_samples(data_dir=data_dir, output_dir=output_dir)
    if args.generated:
        show_generated_results(input_dir=args.input_dir, output_dir=output_dir)
    if not args.samples and not args.generated:
        parser.print_help()
