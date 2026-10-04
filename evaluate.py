"""
验证码生成质量评估工具

用法:
  # 评估最新 checkpoint 的生成质量
  python evaluate.py

  # 指定 checkpoint
  python evaluate.py --checkpoint ./output_v28/checkpoint_epoch_400.pt

  # 生成并评估 N 张图片
  python evaluate.py --count 500

  # 同时评估训练 G (非 EMA) 的生成质量
  python evaluate.py --use-training-g

评估指标:
  - CaptchaResNet 可读性 (权威指标, 打破判别器自证循环)
  - D Aux 准确率: 判别器辅助分类头对生成字符的识别率 (仅参考)
  - 像素统计: 均值、标准差、非零像素比例 (与真实数据对比)
  - 模式坍塌检测: 同标签多样性与非零像素坍塌比例
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from collections import Counter

from config import *
from models import Generator, Discriminator
from device_utils import DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="验证码生成质量评估")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型检查点路径 (默认自动选择最新)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"checkpoint 搜索目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--count", type=int, default=200,
                        help="评估生成的图片数量")
    parser.add_argument("--use-training-g", action="store_true",
                        help="使用训练 G 而非 EMA G 进行评估")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="生成时的 batch 大小")
    parser.add_argument("--output", type=str, default="evaluation",
                        help="评估结果输出目录")
    return parser.parse_args()


def find_best_checkpoint(output_dir: Path) -> Path:
    """自动查找最新的 checkpoint"""
    checkpoints = sorted(output_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"未找到 checkpoint 文件！\n  在 {output_dir} 中未找到 checkpoint_epoch_*.pt")
    return checkpoints[-1]


@torch.no_grad()
def generate_batch(G, labels, z=None):
    """生成一批验证码"""
    B = labels.shape[0]
    if z is None:
        z = torch.randn(B, LATENT_DIM, device=DEVICE)
    fake = G(z, labels)
    return fake


def compute_pixel_stats(images: torch.Tensor) -> dict:
    """计算像素统计量并与真实数据对比"""
    # images: (N, 1, H, W) in tanh range [-1, 1]
    # 反归一化到 [0, 255] 用于统计
    imgs_uint8 = (images.detach().cpu() * 0.5 + 0.5).clamp(0, 1) * 255
    imgs_uint8 = imgs_uint8.numpy().astype(np.uint8)

    N = imgs_uint8.shape[0]
    per_image_mean = imgs_uint8.reshape(N, -1).mean(axis=1)
    per_image_std = imgs_uint8.reshape(N, -1).std(axis=1)
    nonzero_pct = (imgs_uint8 > 10).mean(axis=(1, 2, 3)) * 100

    return {
        "pixel_mean": per_image_mean.mean(),
        "pixel_mean_std": per_image_mean.std(),
        "pixel_std": per_image_std.mean(),
        "pixel_std_std": per_image_std.std(),
        "nonzero_pct": nonzero_pct.mean(),
        "nonzero_pct_std": nonzero_pct.std(),
        # 检测是否坍塌 (非零像素 < 15% 视为坍塌)
        "collapsed_ratio": (nonzero_pct < 15).mean(),
    }


def evaluate_aux_accuracy(D, images, labels) -> dict:
    """使用 D 的 Aux head 评估字符识别准确率"""
    D.eval()
    _, aux, _ = D(images, labels)
    preds = aux.argmax(dim=-1)  # (B, CAPTCHA_LENGTH)
    correct = (preds == labels).float()

    # 整体准确率
    overall_acc = correct.mean().item()
    # 逐位置准确率
    per_pos_acc = correct.mean(dim=0).cpu().tolist()
    # 完全正确率 (4个字符全对)
    perfect_acc = correct.all(dim=1).float().mean().item()

    return {
        "aux_overall_acc": overall_acc,
        "aux_per_pos_acc": per_pos_acc,
        "aux_perfect_acc": perfect_acc,
    }


def evaluate_ocr_accuracy(ocr_model, images, labels) -> dict:
    """用冻结的 CaptchaResNet (真实数据高准确率) 客观评估字形可读性。
    这是权威指标 — 打破判别器自证循环。"""
    import ocr_net
    ocr_model.eval()
    with torch.no_grad():
        logits = ocr_model(images)
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).float()

    overall_acc = correct.mean().item()
    per_pos_acc = correct.mean(dim=0).cpu().tolist()
    perfect_acc = correct.all(dim=1).float().mean().item()
    return {
        "ocr_overall_acc": overall_acc,
        "ocr_per_pos_acc": per_pos_acc,
        "ocr_perfect_acc": perfect_acc,
    }


def evaluate_diversity(G, D, num_samples=100, num_repeats=10):
    """评估生成多样性: 对同一标签生成多次, 检查是否产生不同的图片"""
    # 随机选择几个标签
    labels_pool = torch.randint(0, NUM_CLASSES, (num_samples, CAPTCHA_LENGTH), device=DEVICE)

    # 对每个标签生成 num_repeats 次 (不同噪声)
    all_stds = []
    for i in range(min(num_samples, 20)):  # 取 20 个标签测试
        label = labels_pool[i:i+1].repeat(num_repeats, 1)
        z = torch.randn(num_repeats, LATENT_DIM, device=DEVICE)
        with torch.no_grad():
            fake = G(z, label)
        # 计算这 num_repeats 张图之间的像素级标准差
        pixel_std = fake.std(dim=0).mean().item()
        all_stds.append(pixel_std)

    avg_diversity = np.mean(all_stds)
    return {
        "diversity_score": avg_diversity,
        "diversity_interpretation": (
            "良好" if avg_diversity > 0.15 else
            "一般" if avg_diversity > 0.08 else
            "严重模式坍塌"
        )
    }


def main():
    args = parse_args()
    device = DEVICE
    output_dir = Path(args.output_dir)
    eval_output = Path(args.output)
    eval_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  验证码生成质量评估")
    print(f"{'='*60}")
    print(f"  设备: {DEVICE}")

    # ─── 加载模型 ──────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_best_checkpoint(output_dir)
    print(f"  Checkpoint: {ckpt_path}")

    G = Generator().to(device)
    D = Discriminator().to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    if args.use_training_g:  # 默认使用 EMA G (更稳定)
        # 使用训练 G 权重 — BN running stats 是训练中正确累积的
        G.load_state_dict(ckpt["generator_state_dict"])
        D.load_state_dict(ckpt["discriminator_state_dict"])
        print(f"  [INFO] 使用训练 G 权重 (generator_state_dict)")
        if "ema_state_dict" in ckpt:
            print(f"  [TIP]  checkpoint 含 EMA 权重, 不加 --use-training-g 可加载 EMA")
    else:
        # 使用 EMA 权重
        G.load_state_dict(ckpt.get("ema_state_dict", ckpt["generator_state_dict"]))
        D.load_state_dict(ckpt["discriminator_state_dict"])
        print(f"  [INFO] 使用 EMA Generator 权重")

    G.eval()
    D.eval()

    print(f"\n{'='*60}")
    print(f"  生成 {args.count} 张验证码进行评估...")
    print(f"{'='*60}")

    # ─── 批量生成 ──────────────────────────────────────
    all_fakes = []
    all_labels = []
    remaining = args.count
    with torch.no_grad():
        while remaining > 0:
            B = min(args.batch_size, remaining)
            labels = torch.randint(0, NUM_CLASSES, (B, CAPTCHA_LENGTH), device=device)
            z = torch.randn(B, LATENT_DIM, device=device)
            fake = G(z, labels)
            all_fakes.append(fake.cpu())
            all_labels.append(labels.cpu())
            remaining -= B

    all_fakes = torch.cat(all_fakes, dim=0)[:args.count]
    all_labels = torch.cat(all_labels, dim=0)[:args.count]

    print(f"  生成完成: {all_fakes.shape}")

    # ─── 1. 像素统计 ──────────────────────────────────
    print(f"\n  ─── 像素统计 ───")
    pixel_stats = compute_pixel_stats(all_fakes)
    print(f"    平均亮度:       {pixel_stats['pixel_mean']:.1f} ± {pixel_stats['pixel_mean_std']:.1f}")
    print(f"    平均对比度:     {pixel_stats['pixel_std']:.1f} ± {pixel_stats['pixel_std_std']:.1f}")
    print(f"    非零像素比例:   {pixel_stats['nonzero_pct']:.1f}% ± {pixel_stats['nonzero_pct_std']:.1f}%")
    print(f"    坍塌样本比例:   {pixel_stats['collapsed_ratio']*100:.1f}%")
    print(f"    参考 (真实):    亮度≈181, 对比度≈53, 非零≈96%")

    # ─── 2. 字符识别 ────────────────────────────────
    print(f"\n  ─── 强OCR 字形可读性 (权威指标) ───")
    import ocr_net
    ocr_model = ocr_net.load_ocr(device=device)
    all_fakes_gpu = all_fakes.to(device)
    all_labels_gpu = all_labels.to(device)
    ocr_stats = evaluate_ocr_accuracy(ocr_model, all_fakes_gpu, all_labels_gpu)
    print(f"    逐字符准确率:   {ocr_stats['ocr_overall_acc']*100:.1f}%")
    print(f"    逐位置准确率:   {[f'{a*100:.1f}%' for a in ocr_stats['ocr_per_pos_acc']]}")
    print(f"    完全正确率:     {ocr_stats['ocr_perfect_acc']*100:.1f}%")

    print(f"\n  ─── D Aux 字符识别 (自证, 仅参考) ───")
    aux_stats = evaluate_aux_accuracy(D, all_fakes_gpu, all_labels_gpu)
    print(f"    逐字符准确率:   {aux_stats['aux_overall_acc']*100:.1f}%")
    print(f"    完全正确率:     {aux_stats['aux_perfect_acc']*100:.1f}%")

    # ─── 3. 多样性检测 ────────────────────────────────
    print(f"\n  ─── 生成多样性 ───")
    div_stats = evaluate_diversity(G, D)
    print(f"    多样性分数:     {div_stats['diversity_score']:.4f}")
    print(f"    评估:           {div_stats['diversity_interpretation']}")

    # ─── 4. 保存样本 ──────────────────────────────────
    print(f"\n  ─── 保存样本 ───")
    # 保存一些样本图
    sample_imgs = all_fakes[:64]
    sample_imgs_uint8 = (sample_imgs * 0.5 + 0.5).clamp(0, 1) * 255
    sample_imgs_uint8 = sample_imgs_uint8.numpy().astype(np.uint8)

    # 单独保存前 20 张
    for i in range(min(20, len(sample_imgs_uint8))):
        label_str = "".join(CHARS[all_labels[i, j].item()] for j in range(CAPTCHA_LENGTH))
        img = Image.fromarray(sample_imgs_uint8[i, 0])
        img.save(eval_output / f"eval_{i:03d}_{label_str}.png")

    # 拼接预览图
    grid_imgs = np.concatenate([sample_imgs_uint8[i, 0] for i in range(min(20, len(sample_imgs_uint8)))], axis=1)
    Image.fromarray(grid_imgs).save(eval_output / "eval_grid.png")
    print(f"    样本保存在: {eval_output.resolve()}")

    # ─── 5. 综合评分 ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  综合评估结果")
    print(f"{'='*60}")

    # 评分 (0-100)
    score_pixel = min(100, pixel_stats['nonzero_pct'] * 1.0)  # 目标 ~96%
    score_aux = ocr_stats['ocr_overall_acc'] * 100  # 权威 OCR 指标
    score_anti_collapse = (1.0 - pixel_stats['collapsed_ratio']) * 100
    score_diversity = min(100, div_stats['diversity_score'] / 0.2 * 100)

    total_score = score_pixel * 0.2 + score_aux * 0.35 + score_anti_collapse * 0.25 + score_diversity * 0.2

    issues = []
    if pixel_stats['collapsed_ratio'] > 0.1:
        issues.append(f"WARN: {pixel_stats['collapsed_ratio']*100:.0f}% samples collapsed (near-black/blank)")
    if ocr_stats['ocr_overall_acc'] < 0.3:
        issues.append(f"WARN: Character recognition too low ({ocr_stats['ocr_overall_acc']*100:.1f}%)")
    if div_stats['diversity_score'] < 0.08:
        issues.append(f"WARN: Severe mode collapse (diversity={div_stats['diversity_score']:.4f})")
    if pixel_stats['pixel_std'] < 20:
        issues.append(f"WARN: Contrast too low ({pixel_stats['pixel_std']:.1f})")

    print(f"  像素质量:       {score_pixel:.0f}/100")
    print(f"  字符辨识:       {score_aux:.0f}/100")
    print(f"  抗坍塌:         {score_anti_collapse:.0f}/100")
    print(f"  多样性:         {score_diversity:.0f}/100")
    print(f"  ─────────────────────")
    print(f"  综合评分:       {total_score:.0f}/100")

    if issues:
        print(f"\n  Issues found:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  [OK] No significant issues found")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
