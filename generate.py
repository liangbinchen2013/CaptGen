"""
生成验证码

用法:
  python generate.py                                               # 随机生成，输出到 ./generated/
  python generate.py --count 100                                   # 生成 100 张
  python generate.py --text "ab12" "xy78"                          # 指定文本
  python generate.py --checkpoint ./output/checkpoint_epoch_300.pt # 指定模型
  python generate.py --checkpoint ./output/epoch_300.pt --output ./my_captchas  # 指定模型和输出目录
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from config import *
from models import Generator
from utils import load_model
from device_utils import DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="生成验证码")
    parser.add_argument("--checkpoint", type=str,
                        default=None,
                        help="模型检查点路径 (默认自动选择 OUTPUT_DIR 中最新的)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"训练时 checkpoint 存放目录 (默认: {OUTPUT_DIR})，用于自动查找最新模型")
    parser.add_argument("--output", type=str, default="generated",
                        help="生成图片输出目录")
    parser.add_argument("--count", type=int, default=10,
                        help="随机生成数量")
    parser.add_argument("--label-dist", choices=["real", "uniform"], default="real",
                        help="随机标签分布: real=按真实数据分布(默认), uniform=36类均匀")
    parser.add_argument("--text", type=str, nargs="+", default=None,
                        help="指定要生成的文本 (如 --text ab12 xy78)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def text_to_label(text: str) -> torch.LongTensor:
    """将 'ab12' → tensor([idx_a, idx_b, idx_1, idx_2])
    自动应用 LABEL_CASE_MAP 修正大小写
    """
    # 修正大小写 (o→O 等视觉不可区分的字符)
    text = "".join(LABEL_CASE_MAP.get(c, c) for c in text)
    ids = []
    for c in text:
        if c not in CHARS:
            raise ValueError(f"字符 '{c}' 不在字符集中 (可用: {CHARS})")
        ids.append(CHARS.index(c))
    return torch.tensor(ids, dtype=torch.long)


def load_label_counts():
    """读取 confusion_profile.json 中的真实标签分布 (部分字符稀有或缺失)"""
    profile = Path(__file__).resolve().parent / "confusion_profile.json"
    try:
        import json
        counts = json.loads(profile.read_text(encoding="utf-8")).get("label_counts", {})
        counts = {c: int(v) for c, v in counts.items() if c in CHARS and int(v) > 0}
        return counts or None
    except Exception:
        return None


def sample_labels_real(count: int) -> torch.LongTensor:
    """按真实标签分布采样 — 与训练分布一致 (默认)

    均匀采样会生成训练集中从未出现的字符, 模型对该标签没有学习信号,
    输出质量不可控 — 因此默认按真实分布采样。
    """
    counts = load_label_counts()
    if not counts:
        print("[WARN] 未找到 confusion_profile.json, 回退均匀采样 "
              "(注意: 字符 '0' 在训练集中从未出现)")
        return torch.randint(0, NUM_CLASSES, (count, CAPTCHA_LENGTH))
    chars = list(counts.keys())
    w = np.array([counts[c] for c in chars], dtype=np.float64)
    w /= w.sum()
    idx = np.random.choice(len(chars), size=(count, CAPTCHA_LENGTH), p=w)
    return torch.tensor([[CHARS.index(chars[j]) for j in row] for row in idx],
                        dtype=torch.long)


def find_best_checkpoint() -> Path:
    """自动查找最新的 checkpoint"""
    checkpoints = sorted(OUTPUT_DIR.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"未找到 checkpoint 文件！请先训练: python train.py\n"
            f"  或在 {OUTPUT_DIR} 中放置 .pt 文件"
        )
    return checkpoints[-1]


@torch.no_grad()
def generate():
    args = parse_args()
    device = DEVICE

    # ─── 路径 ──────────────────────────────────────────
    # 如果指定了 output-dir，覆盖 config 的默认 OUTPUT_DIR
    # 这样 find_best_checkpoint() 会在正确目录查找
    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output_dir)
    print(f"[PATH] checkpoint 搜索目录: {OUTPUT_DIR}")

    # ─── 加载模型 ──────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_best_checkpoint()

    print(f"[LOAD] 加载模型: {ckpt_path}")
    generator = Generator().to(device).eval()
    load_model(ckpt_path, generator, None, device=device)
    print(f"[OK] 模型加载完成")

    # ─── 准备标签 ──────────────────────────────────────
    if args.text:
        texts = args.text
        labels_list = [text_to_label(t) for t in texts]
        labels = torch.stack(labels_list).to(device)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        texts = None
        if args.label_dist == "real":
            labels = sample_labels_real(args.count).to(device)
        else:
            labels = torch.randint(0, NUM_CLASSES, (args.count, CAPTCHA_LENGTH)).to(device)

    # ─── 生成 ──────────────────────────────────────────
    z = torch.randn(labels.shape[0], LATENT_DIM, device=device)
    fake = generator(z, labels)

    # 反归一化 [−1,1] → [0,255]
    fake = (fake.cpu() * 0.5 + 0.5).clamp(0, 1) * 255
    fake = fake.numpy().astype(np.uint8)

    # ─── 保存 ──────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[OK] 生成 {len(fake)} 张验证码:")
    for i in range(len(fake)):
        if texts:
            label = texts[i]
        else:
            label = "".join(CHARS[labels[i, j].item()] for j in range(CAPTCHA_LENGTH))

        img = fake[i, 0]  # (H, W)
        path = output_dir / f"gen_{label}.png"
        Image.fromarray(img).save(path)
        print(f"  [{i+1}/{len(fake)}] {label} → {path}")

    # 拼接成一张大图方便查看
    grid_path = output_dir / "grid.png"
    grid_img = np.concatenate([fake[i, 0] for i in range(min(len(fake), 20))], axis=1)
    Image.fromarray(grid_img).save(grid_path)
    print(f"\n  [GRID] 拼接预览: {grid_path}")
    print(f"\n[OK] 完成！文件保存在 {output_dir.resolve()}")


if __name__ == "__main__":
    generate()
