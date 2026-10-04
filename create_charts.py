"""
训练曲线绘图工具

从训练日志 (training_log.csv) 生成 README / 发布所需的图表:
  docs/loss_curves.png    G/D/Aux/FM 损失 + 正则项 + 学习率
  docs/ocr_curves.png     OCR 准确率 / gap 墨迹比 / 判别器平衡

用法:
  python create_charts.py
  python create_charts.py --csv output_v28/training_log.csv --out docs
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体回退 (Windows/Linux 常见字体)
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _smooth(y, window=51):
    """滑动平均, 用于 step 级噪声曲线"""
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().values


def _mark_phase_c(ax, epoch):
    """标记 Phase C 起点 (混淆软目标)"""
    if epoch is not None and epoch > 0:
        ax.axvline(epoch, color="gray", ls="--", lw=1.0, alpha=0.7)
        ax.text(epoch, ax.get_ylim()[1], " Phase C", color="gray",
                fontsize=8, va="top", ha="left")


def plot_loss_curves(df: pd.DataFrame, out: Path, phase_c_epoch: int = 300):
    steps = df[df["phase"] != "EPOCH"]
    steps = steps[~steps["phase"].str.endswith("_EPOCH", na=False)]
    x = steps["step"].values

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=120)

    ax = axes[0, 0]
    for col, label in [("d_loss", "D Loss"), ("g_loss", "G Loss"),
                       ("aux_loss", "Aux Loss"), ("fm_loss", "FM Loss")]:
        ax.plot(x, _smooth(steps[col].values), label=label, lw=1.1)
    ax.set_title("GAN loss components (smoothed)")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for col, label in [("edge_loss", "Edge Loss"), ("realism_loss", "Realism Loss"),
                       ("div_loss", "Diversity Loss"), ("contrast_loss", "Contrast Loss")]:
        ax.plot(x, _smooth(steps[col].values), label=label, lw=1.1)
    ax.set_title("Visual quality regularizers (smoothed)")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, _smooth(steps["r1_penalty"].values), label="R1 penalty", lw=1.0)
    ax.plot(x, _smooth(steps["r2_penalty"].values), label="R2 penalty", lw=1.0)
    ax.plot(x, _smooth(steps["g_grad_norm"].values), label="G grad norm", lw=1.0, alpha=0.8)
    ax.plot(x, _smooth(steps["d_grad_norm"].values), label="D grad norm", lw=1.0, alpha=0.8)
    ax.set_title("Regularization / gradient norms (smoothed)")
    ax.set_xlabel("Step"); ax.set_ylabel("Value")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, steps["lr_g"].values, label="LR G", lw=1.2)
    ax.plot(x, steps["lr_d"].values, label="LR D", lw=1.2, alpha=0.8)
    ax.set_title("Learning rate schedule")
    ax.set_xlabel("Step"); ax.set_ylabel("LR")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    path = out / "loss_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] {path}")


def plot_ocr_curves(df: pd.DataFrame, out: Path, phase_c_epoch: int = 300):
    ep = df[df["phase"].str.endswith("_EPOCH", na=False)]
    x = ep["epoch"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

    ax = axes[0]
    ax.plot(x, ep["ocr_acc"].values, label="CaptchaResNet", lw=1.4)
    dddd = ep["dddd_acc"].astype(float).replace(-1.0, np.nan)
    ppll = ep["ppll_acc"].astype(float).replace(-1.0, np.nan)
    ax.plot(x, dddd.values, label="ddddocr", lw=1.4)
    ax.plot(x, ppll.values, label="ppllocr", lw=1.4)
    ax.axhline(0.7983, color="tab:orange", ls=":", lw=1.2,
               label="ddddocr real 79.8%")
    ax.axhline(0.6787, color="tab:green", ls=":", lw=1.2,
               label="ppllocr real 67.9%")
    _mark_phase_c(ax, phase_c_epoch)
    ax.set_title("OCR accuracy (whole-string)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(x, ep["gap_ink"].values, label="gap ink ratio", lw=1.4)
    real_gap = np.mean([0.836, 0.962, 0.693])
    ax.axhline(real_gap, color="gray", ls=":", lw=1.2,
               label=f"real mean {real_gap:.2f}")
    _mark_phase_c(ax, phase_c_epoch)
    ax.set_title("Character gap ink ratio (anti-adhesion)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Ratio")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(x, ep["d_real_mean"].astype(float).values, label="D(real)", lw=1.3)
    ax.plot(x, ep["d_fake_mean"].astype(float).values, label="D(fake)", lw=1.3)
    gap = ep["d_real_mean"].astype(float).values - ep["d_fake_mean"].astype(float).values
    ax.plot(x, gap, label="D gap", lw=1.3, color="black", alpha=0.7)
    _mark_phase_c(ax, phase_c_epoch)
    ax.set_title("Discriminator balance")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    path = out / "ocr_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path("output_v28") / "training_log.csv"))
    ap.add_argument("--out", default="docs")
    ap.add_argument("--phase-c-epoch", type=int, default=300)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到训练日志: {csv_path}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"[INFO] 日志: {csv_path} ({len(df)} 行, epoch {df['epoch'].min()}~{df['epoch'].max()})")

    plot_loss_curves(df, out, args.phase_c_epoch)
    plot_ocr_curves(df, out, args.phase_c_epoch)


if __name__ == "__main__":
    main()
