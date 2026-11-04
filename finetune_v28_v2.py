# -*- coding: utf-8 -*-
"""
v28 微调 v2.1 — 最终选定配方: 5 epoch (15-epoch cosine 调度), 从 v28 EMA 起步。

选定依据 (上一轮 15ep 多 seed 评测):
  - ep5 是唯一同时满足 "三 seed 全过 M1/M2/M4/M6/M8" 且 M9 有改善的点:
    M9 0.2054 vs v28 0.2188; ppll M1 0.8895 vs 0.8900 (不被推坏);
    dddd M2 0.9566 / M4 0.9270 (更接近真实); M8 0.9618.
  - ep10/15 ppll 开始漂移 (M1 0.881/0.872, M2ppll 1.241/1.253,
    M6 0.047/0.049; ep15 s43 M6 0.0522 FAIL) -> 不再继续训练.
  - 像素漂移: v2 ep1->15 字符区 mean|diff| 11.7/p99 100; ep1->5 7.6/p99 64;
    同权重噪声底仅 0.27/p99 3 -> ep5 之后的漂移是真实函数变化, 足以伤 ppll.

v2 核心机制 (保持):
  1. G 从 base 的 EMA 初始化; D 从 base; EMA 同步.
  2. 冻结 G/D 的 BN running stats; ckpt 在 sample 之前保存.
  3. per-OCR adaptive ε (默认 merged): ε_c=max(0, real_rate_c - gen_rate_c);
     过混淆字符 ε=0 -> 硬 CE 纠错; 欠混淆字符 O/L/1/C/S/V 补混淆.
  4. ppll pair-hard CE (覆盖率有限, 保留).
  5. 多 eval seed (41/42/43) guard; best 按平均 M9; 任一 seed 不过则不选 best.
  6. 默认 5 epoch, 但 cosine T_max=15 (与选定 run 的 ep5 学习率一致, 可复现).

用法:
  # 前置: 先完成 v28 训练, 得到 output_v28/checkpoint_epoch_400.pt
  #       以及 output_v28/gen_error_stats_{ddddocr,ppllocr}_ep400.txt
  python finetune_v28_v2.py                 # 5ep 选定配方, 输出 output_v2_5ep/
  python finetune_v28_v2.py --epochs 5 --sched-epochs 15
  python finetune_v28_v2.py --output-dir output_v2_5ep_repro

  # 自定义基座/数据/统计文件位置
  python finetune_v28_v2.py --base-ckpt path/to/checkpoint_epoch_400.pt \
      --data-dir path/to/训练数据 --eps-stats-prefix path/to/gen_error_stats

  # 对照: 只继续训练 5ep
  python finetune_v28_v2.py --eps-mode none --lambda-soft 0 --pair-hard-weight 0
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    CHARS, NUM_CLASSES, CAPTCHA_LENGTH, LATENT_DIM,
    CONFUSION_DIST, OCR_ERROR_RATE, EPS_CAP,
    LABEL_SMOOTHING_EPS, EMA_DECAY, CLIP_NORM,
    CONTRAST_MARGIN, CONTRAST_LO_Q, CONTRAST_HI_Q,
    EDGE_LOSS_MIN, SEED,
)
from models import Generator, Discriminator, EMAGenerator
from data_loader import get_dataloader
from utils import save_sample
from adaptive_ocr import MultiOCRAdaptive
import ocr_net
from device_utils import (
    DEVICE, device_supports_pin_memory, autocast, get_scaler,
    scaler_unscale_, use_amp, empty_cache, manual_seed_all,
)
from train import (
    ADAAugment, edge_loss, realism_loss, diversity_loss,
    feature_matching_loss, compute_r1_penalty, compute_r2_penalty,
    soft_ce, clip_gradients, _batch_quantile,
)


def sample_real_labels(n: int, device):
    """按真实标签分布采样固定样例标签 (排除训练集从未出现的 '0')."""
    try:
        p = ROOT / "confusion_profile.json"
        counts = json.loads(p.read_text(encoding="utf-8")).get("label_counts", {})
        chars = [c for c in CHARS if counts.get(c, 0) > 0]
        if chars:
            w = torch.tensor([counts[c] for c in chars], dtype=torch.float32)
            w = w / w.sum()
            idx = torch.multinomial(w, n * CAPTCHA_LENGTH, replacement=True)
            idx = idx.view(n, CAPTCHA_LENGTH)
            char_ids = torch.tensor([CHARS.index(c) for c in chars], dtype=torch.long)
            return char_ids[idx].to(device)
    except Exception as e:
        print(f"  [WARN] 真实标签分布采样失败, 回退排除 '0': {e}")
    return torch.randint(1, NUM_CLASSES, (n, CAPTCHA_LENGTH), device=device)


# v28 权重
V28_LAMBDA_AUX_G = 1.5
V28_LAMBDA_AUX_D = 1.0
V28_LAMBDA_FM = 1.0
V28_LAMBDA_CONTRAST = 3.0
V28_LAMBDA_DIVERSITY = 1.0
V28_LAMBDA_EDGE = 2.0
V28_LAMBDA_REALISM = 1.0
V28_LAMBDA_R1 = 0.6
V28_LAMBDA_R2 = 0.6
V28_ADV_LAMBDA = 0.55

# v29 实证超额对 + v28/v2 分析补充 (I->1/I->E/N->M/M->T/A->4 ...)
PAIR_LIST = (
    ("I", "H"), ("R", "T"), ("R", "X"), ("R", "F"), ("K", "T"), ("W", "T"),
    ("7", "T"), ("M", "T"), ("N", "H"), ("N", "A"), ("4", "A"), ("U", "B"),
    ("G", "C"), ("G", "E"), ("G", "O"), ("Q", "9"), ("I", "J"), ("I", "T"),
    ("V", "T"), ("L", "Z"), ("Z", "2"),
    ("I", "X"), ("R", "J"), ("R", "N"), ("R", "W"), ("R", "8"), ("T", "L"),
    ("A", "E"), ("T", "F"), ("Z", "R"), ("9", "T"), ("D", "P"), ("D", "J"),
    ("M", "N"), ("7", "1"), ("4", "9"),
    # v2 补充: 上一轮 ep30 的头部对 / 逐字符超额
    ("I", "1"), ("I", "E"), ("I", "L"), ("N", "M"), ("M", "T"), ("A", "4"),
    ("G", "9"), ("R", "C"), ("R", "7"), ("F", "E"), ("H", "F"), ("W", "V"),
)
PAIR_SETS = defaultdict(set)
for _t, _p in PAIR_LIST:
    if _t in CHARS and _p in CHARS:
        PAIR_SETS[_t].add(_p)


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    manual_seed_all(seed)
    print(f"  [SEED] {seed}")


def freeze_bn(model):
    """冻结 BN running stats; 保留 affine 参数可训练."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = 0.0
            n += 1
    return n


def build_real_dist():
    """完整真实混淆分布 (不抑制); ε=0 的字符在 soft_ce 中退化为硬 CE."""
    dist = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    for ch, row in CONFUSION_DIST.items():
        ci = CHARS.index(ch)
        for p, v in row.items():
            if p in CHARS:
                dist[ci, CHARS.index(p)] = v
    return dist


_PAIR_RE = re.compile(r'^(<LEN>|[0-9A-Z])->(\S*)\s+(\d+)')


def _parse_errs(path: Path):
    errs = defaultdict(int)
    if not path.exists():
        return None
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = _PAIR_RE.match(ln.strip())
        if not m:
            continue
        t, c = m.group(1), int(m.group(3))
        if t != "<LEN>":
            errs[t] += c
    return errs


def build_eps_from_stats(prefix: str, tag: str, n: int, seed: int,
                         mode: str, cap: float):
    """从 v28 的 gen_error_stats 计算 per-OCR adaptive ε.

    mode: merged (dddd+ppll), ppll, dddd, static, none.
    ε_c = clamp(real_rate_c - gen_rate_c, 0, cap); 过混淆字符 ε=0 -> 硬 CE.
    """
    profile = json.loads((ROOT / "confusion_profile.json").read_text(encoding="utf-8"))
    if mode in ("static", "none"):
        eps = torch.zeros(NUM_CLASSES)
        if mode == "static":
            for ch, v in profile.get("error_rate", {}).items():
                if ch in CHARS:
                    eps[CHARS.index(ch)] = min(float(v), cap)
        return eps, {"mode": mode, "source": "static"}

    base = Path(prefix)
    d_path = Path(f"{base}_ddddocr_{tag}.txt")
    p_path = Path(f"{base}_ppllocr_{tag}.txt")
    D = _parse_errs(d_path)
    P = _parse_errs(p_path)
    if (mode == "dddd" and D is None) or (mode == "ppll" and P is None) or \
            (mode == "merged" and D is None and P is None):
        print(f"  [WARN] eps stats 缺失 ({d_path} / {p_path}), 回退 static")
        eps = torch.zeros(NUM_CLASSES)
        for ch, v in profile.get("error_rate", {}).items():
            if ch in CHARS:
                eps[CHARS.index(ch)] = min(float(v), cap)
        return eps, {"mode": "static_fallback", "source": "static"}

    # 复现 eval 的 label 采样 (evaluate_generated.main: np.random.seed(seed))
    counts = profile["label_counts"]
    chars = [c for c in CHARS if counts.get(c, 0) > 0]
    w = np.array([counts[c] for c in chars], dtype=np.float64)
    w /= w.sum()
    np.random.seed(seed)
    idx = np.random.choice(len(chars), size=(n, CAPTCHA_LENGTH), p=w)
    occ = defaultdict(int)
    for row in idx:
        for j in row:
            occ[chars[j]] += 1

    eps = torch.zeros(NUM_CLASSES)
    used = []
    for ch in CHARS:
        if occ.get(ch, 0) == 0:
            continue
        rr = float(profile.get("error_rate", {}).get(ch, 0.0))
        if mode == "dddd":
            err, denom = (D or {}).get(ch, 0), occ[ch]
        elif mode == "ppll":
            err, denom = (P or {}).get(ch, 0), occ[ch]
        else:
            err = (D or {}).get(ch, 0) + (P or {}).get(ch, 0)
            denom = occ[ch] * 2
        gr = err / max(denom, 1)
        e = min(max(0.0, rr - gr), cap)
        eps[CHARS.index(ch)] = e
        if e > 0.01:
            used.append((e, ch, rr, gr))
    used.sort(reverse=True)
    info = {"mode": mode, "source": f"{d_path.name}|{p_path.name}",
            "eps_sum": float(eps.sum()),
            "top": [(ch, round(e, 3), round(rr, 3), round(gr, 3))
                    for e, ch, rr, gr in used[:12]]}
    print(f"  [EPS] mode={mode} sum={float(eps.sum()):.3f} "
          f"top={info['top'][:6]}")
    return eps, info


def parse_report(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")

    def grab(pat):
        m = re.search(pat, text)
        return float(m.group(1)) if m else None

    return {
        "m1_dddd": grab(r"M1 ddddocr acc比 fake/real:\s*([0-9.]+)"),
        "m1_ppll": grab(r"M1 ppllocr acc比 fake/real:\s*([0-9.]+)"),
        "m2_dddd": grab(r"M2 ddddocr 字符错误率比:\s*([0-9.]+)"),
        "m2_ppll": grab(r"M2 ppllocr 字符错误率比:\s*([0-9.]+)"),
        "m3_excess": grab(r"M3 混淆JS超额.*?:\s*([\-0-9.]+)"),
        "m4_dddd": grab(r"M4 ddddocr <LEN>错误比.*?:\s*([0-9.]+)"),
        "m4_ppll": grab(r"M4 ppllocr <LEN>错误比.*?:\s*([0-9.]+)"),
        "m6": grab(r"M6 逐字符错误率MAE:\s*([0-9.]+)"),
        "m8": grab(r"M8 CaptchaResNet整图acc \(代理\):\s*([0-9.]+)"),
        "m9": grab(r"M9 归一化超额top10和:\s*([0-9.]+)"),
    }


def _mean(results, key):
    vals = [m[key] for _, m in results if m.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _seed_ok(m, args):
    if not m:
        return False
    req = [m.get("m1_dddd"), m.get("m1_ppll"), m.get("m2_ppll"),
           m.get("m6"), m.get("m8"), m.get("m2_dddd"),
           m.get("m4_dddd"), m.get("m4_ppll")]
    if any(v is None for v in req):
        return False
    return (m["m1_dddd"] >= args.guard_m1 and m["m1_ppll"] >= args.guard_m1
            and m["m2_dddd"] >= 0.8 and m["m2_ppll"] <= args.guard_m2_ppll
            and m["m6"] <= args.guard_m6 and m["m8"] >= 0.85
            and 0.5 <= m["m4_dddd"] <= 2.0
            and 0.5 <= m["m4_ppll"] <= 2.0)


def main():
    ap = argparse.ArgumentParser(description="v28 微调 v2 (EMA/BN/adaptive eps/pair-hard)")
    ap.add_argument("--base-ckpt", default=str(ROOT / "output_v28" / "checkpoint_epoch_400.pt"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "训练数据"))
    ap.add_argument("--output-dir", default=str(ROOT / "output_v2_5ep"))
    ap.add_argument("--epochs", type=int, default=5,
                    help="实际训练 epoch 数 (选定配方=5)")
    ap.add_argument("--sched-epochs", type=int, default=15,
                    help="Cosine T_max; 默认 15 以复现选定 run 的 ep5 学习率")
    ap.add_argument("--final-name", default="checkpoint_v2_5ep.pt",
                    help="训练结束后 best/last 的稳定别名; 空字符串=不生成")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-min", type=float, default=5e-6)
    ap.add_argument("--d-updates", type=int, default=1)
    ap.add_argument("--clip-norm", type=float, default=CLIP_NORM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ada-p", type=float, default=1.0)
    ap.add_argument("--ema-init", action="store_true", default=True,
                    help="G 从 base 的 EMA 初始化 (默认开)")
    ap.add_argument("--no-ema-init", action="store_false", dest="ema_init")
    ap.add_argument("--bn-freeze", action="store_true", default=True)
    ap.add_argument("--no-bn-freeze", action="store_false", dest="bn_freeze")
    ap.add_argument("--eps-mode", choices=["merged", "ppll", "dddd", "static", "none"],
                    default="merged")
    ap.add_argument("--eps-stats-prefix", default=str(ROOT / "output_v28" / "gen_error_stats"))
    ap.add_argument("--eps-tag", default="ep400")
    ap.add_argument("--eps-n", type=int, default=2000)
    ap.add_argument("--eps-seed", type=int, default=42)
    ap.add_argument("--eps-cap", type=float, default=EPS_CAP)
    ap.add_argument("--lambda-soft", type=float, default=0.1)
    ap.add_argument("--lambda-feat", type=float, default=0.02)
    ap.add_argument("--pair-hard-weight", type=float, default=0.5)
    ap.add_argument("--pair-from", choices=["ppll", "dddd", "both"], default="ppll")
    ap.add_argument("--eval-interval", type=int, default=5)
    ap.add_argument("--eval-seeds", default="41,42,43")
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--guard-m1", type=float, default=0.85)
    ap.add_argument("--guard-m2-ppll", type=float, default=1.3)
    ap.add_argument("--guard-m6", type=float, default=0.05)
    ap.add_argument("--no-stop-on-guard", action="store_true")
    ap.add_argument("--sample-interval", type=int, default=5)
    ap.add_argument("--save-interval", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    args.d_updates = max(1, int(args.d_updates))
    sched_epochs = int(args.sched_epochs) if int(args.sched_epochs) > 0 else args.epochs
    eval_seeds = [int(s) for s in str(args.eval_seeds).split(",") if s.strip()]

    seed_everything(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    print(f"  [FT2] base={args.base_ckpt} out={outdir}")
    print(f"  [FT2] ema_init={args.ema_init} bn_freeze={args.bn_freeze} "
          f"eps={args.eps_mode} soft={args.lambda_soft} feat={args.lambda_feat} "
          f"pair_hard={args.pair_hard_weight}({args.pair_from})")
    print(f"  [FT2] lr={args.lr} epochs={args.epochs} sched_epochs={sched_epochs} "
          f"eval_seeds={eval_seeds}")

    G = Generator().to(device)
    D = Discriminator().to(device)
    ema_g = EMAGenerator(G, decay=EMA_DECAY).to(device)
    g_optim = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    d_optim = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
    g_sched = optim.lr_scheduler.CosineAnnealingLR(g_optim, T_max=sched_epochs,
                                                  eta_min=args.lr_min)
    d_sched = optim.lr_scheduler.CosineAnnealingLR(d_optim, T_max=sched_epochs,
                                                  eta_min=args.lr_min)
    aux_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING_EPS)
    ada = ADAAugment(target=0.6, p_inc=2e-4, p_dec=5e-5, adjust_interval=4)
    ada.p = float(args.ada_p)
    scaler = get_scaler() if use_amp() else None
    use_amp_flag = scaler is not None

    start_epoch = 1
    global_step = 0
    best_mean_m9 = float("inf")
    best_epoch = -1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        G.load_state_dict(ckpt["generator_state_dict"])
        D.load_state_dict(ckpt["discriminator_state_dict"])
        ema_g.load_state_dict(ckpt["ema_state_dict"])
        g_optim.load_state_dict(ckpt["g_optimizer_state_dict"])
        d_optim.load_state_dict(ckpt["d_optimizer_state_dict"])
        if "g_scheduler_state_dict" in ckpt:
            g_sched.load_state_dict(ckpt["g_scheduler_state_dict"])
            d_sched.load_state_dict(ckpt["d_scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        ada.p = float(ckpt.get("ada_p", args.ada_p))
        best_mean_m9 = float(ckpt.get("best_mean_m9", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", -1))
        print(f"  [FT2] resume ep{start_epoch - 1}, step={global_step}")
    else:
        base = torch.load(args.base_ckpt, map_location=device, weights_only=True)
        if args.ema_init and "ema_state_dict" in base:
            G.load_state_dict(base["ema_state_dict"])
            print("  [FT2] G init from base EMA (clean)")
        else:
            G.load_state_dict(base["generator_state_dict"])
            print("  [FT2] G init from base generator (not EMA)")
        D.load_state_dict(base["discriminator_state_dict"])
        if "ema_state_dict" in base:
            ema_g.load_state_dict(base["ema_state_dict"])
        else:
            ema_g.update(G)
        print(f"  [FT2] loaded base epoch={base.get('epoch')} step={base.get('global_step')}")
    if args.bn_freeze:
        print(f"  [FT2] BN frozen: G={freeze_bn(G)} D={freeze_bn(D)}")

    loader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    pin_memory = device_supports_pin_memory()
    fixed_z = torch.randn(16, LATENT_DIM, device=device)
    fixed_labels = sample_real_labels(16, device)
    fixed_z2 = torch.randn(16, LATENT_DIM, device=device)
    fixed_labels2 = sample_real_labels(16, device)

    adaptive = MultiOCRAdaptive(device=device, load_third_party=True)
    ocr_model = adaptive.ocr_models.get("captcha_resnet")
    if ocr_model is None:
        raise RuntimeError("CaptchaResNet 未加载")

    dist_full = build_real_dist().to(device)
    eps_vec, eps_info = build_eps_from_stats(
        args.eps_stats_prefix, args.eps_tag, args.eps_n, args.eps_seed,
        args.eps_mode, args.eps_cap)
    eps_vec = eps_vec.to(device)
    (outdir / "eps_info.json").write_text(
        json.dumps(eps_info, ensure_ascii=False, indent=1), encoding="utf-8")

    log_path = outdir / "finetune2_log.csv"
    log_new = not log_path.exists() or log_path.stat().st_size == 0
    log_f = open(log_path, "a", encoding="utf-8", buffering=1, newline="")
    log_w = csv.writer(log_f)
    if log_new:
        log_w.writerow(["epoch", "step", "d_loss", "g_loss", "aux", "fm",
                        "contrast", "div", "edge", "realism", "ocr_soft",
                        "ocr_feat", "pair_hard", "pair_ratio", "ocr_w",
                        "ocr_acc", "d_real", "d_fake", "d_gap", "edge_lambda",
                        "eps_mode", "eps_sum", "epoch_time_s"])
    eval_path = outdir / "finetune2_eval.csv"
    eval_new = not eval_path.exists() or eval_path.stat().st_size == 0
    eval_f = open(eval_path, "a", encoding="utf-8", buffering=1, newline="")
    eval_w = csv.writer(eval_f)
    if eval_new:
        eval_w.writerow(["epoch", "m1_dddd", "m1_ppll", "m2_dddd", "m2_ppll",
                         "m3_excess", "m4_dddd", "m4_ppll", "m6", "m8",
                         "m9_mean", "m9_min", "m9_max", "all_seed_ok",
                         "best_mean_m9"])

    stop_after_epoch = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        e_start = time.time()
        sums = {k: 0.0 for k in ("d", "g", "aux", "fm", "ctr", "div", "edge",
                                 "real", "soft", "feat", "pair", "pair_r",
                                 "w", "acc", "eg")}
        n_batches = 0
        for batch_idx, (real_imgs, labels) in enumerate(loader):
            B = real_imgs.size(0)
            real_imgs = real_imgs.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            # ─── D ─────────────────────────────────────
            cached_real_features = None
            _ok = True
            for d_iter in range(args.d_updates):
                d_optim.zero_grad()
                z = torch.randn(B, LATENT_DIM, device=device)
                with torch.no_grad():
                    fake_imgs = G(z, labels)
                need_r1 = (d_iter == args.d_updates - 1)
                if need_r1:
                    real_imgs.requires_grad_(True)
                with autocast():
                    _real = ada.augment(real_imgs, ada.p) if ada.p > 0 else real_imgs
                    _fake = ada.augment(fake_imgs.detach(), ada.p) if ada.p > 0 else fake_imgs.detach()
                    d_real, aux_real, cached_real_features = D(_real, labels, return_features=True)
                    d_fake, aux_fake, _ = D(_fake, labels)
                    d_adv = (F.relu(1.0 + d_fake).mean() +
                             F.relu(1.0 - d_real).mean())
                    aux_loss_real = aux_criterion(aux_real.view(-1, NUM_CLASSES), labels.view(-1))
                    aux_loss_fake = aux_criterion(aux_fake.view(-1, NUM_CLASSES), labels.view(-1))
                    aux_loss_avg = (aux_loss_real + aux_loss_fake) / 2
                    output_reg = 1e-4 * (d_real.mean() ** 2 + d_fake.mean() ** 2)
                if need_r1 and V28_LAMBDA_R1 > 0:
                    r1_val = compute_r1_penalty(d_real.float(), real_imgs)
                else:
                    r1_val = torch.tensor(0.0, device=device)
                if need_r1 and V28_LAMBDA_R2 > 0:
                    fake_imgs.requires_grad_(True)
                    with autocast():
                        d_fake_r2, _, _ = D(fake_imgs, labels)
                    r2_val = compute_r2_penalty(d_fake_r2.float(), fake_imgs)
                    fake_imgs.requires_grad_(False)
                else:
                    r2_val = torch.tensor(0.0, device=device)
                d_loss = (d_adv + V28_LAMBDA_AUX_D * aux_loss_avg +
                          V28_LAMBDA_R1 * r1_val + V28_LAMBDA_R2 * r2_val +
                          output_reg)
                if use_amp_flag:
                    scaler.scale(d_loss).backward()
                    scaler_unscale_(scaler, d_optim)
                else:
                    d_loss.backward()
                _ok = all(p.grad is None or torch.isfinite(p.grad).all()
                          for p in D.parameters() if p.grad is not None)
                if not _ok:
                    print(f"  [WARN] D grad NaN ep{epoch} step{global_step}, skip")
                    d_optim.zero_grad()
                    if use_amp_flag:
                        scaler.update()
                    break
                clip_gradients(D, args.clip_norm)
                if use_amp_flag:
                    scaler.step(d_optim)
                    scaler.update()
                else:
                    d_optim.step()
                ada.update_p(d_real.mean().detach().item())
                real_imgs.requires_grad_(False)
            if not _ok:
                g_optim.zero_grad()
                continue

            # ─── G ─────────────────────────────────────
            g_optim.zero_grad()
            z = torch.randn(B, LATENT_DIM, device=device)
            z2 = torch.randn(max(1, B // 2), LATENT_DIM, device=device)
            with autocast():
                fake_imgs = G(z, labels)
                fake_imgs2 = G(z2, labels[:max(1, B // 2)])
                d_fake, aux_fake, fake_features = D(fake_imgs, labels, return_features=True)

                fake_std = fake_imgs.view(B, -1).std(dim=1)
                with torch.no_grad():
                    real_std = real_imgs.detach().view(B, -1).std(dim=1)
                    _lo = _batch_quantile(real_std, CONTRAST_LO_Q) - CONTRAST_MARGIN
                    _hi = _batch_quantile(real_std, CONTRAST_HI_Q) + CONTRAST_MARGIN
                contrast_loss = (F.relu(_lo - fake_std) + F.relu(fake_std - _hi)).mean()

                cur_edge = edge_loss(fake_imgs, real_imgs)
                cur_realism = realism_loss(fake_imgs, real_imgs)
                cur_div = diversity_loss(fake_imgs[:max(1, B // 2)], fake_imgs2)
                edge_lambda = V28_LAMBDA_EDGE
                if cur_edge.item() < EDGE_LOSS_MIN:
                    edge_lambda = V28_LAMBDA_EDGE * 2.0
                fm_loss = (feature_matching_loss(cached_real_features, fake_features)
                           if cached_real_features is not None
                           else torch.tensor(0.0, device=device))
                adv_loss = -d_fake.mean() * V28_ADV_LAMBDA
                aux_loss = aux_criterion(aux_fake.view(-1, NUM_CLASSES), labels.view(-1))

                # OCR: adaptive ε 的软目标 + ppll pair-hard
                ocr_logits, ocr_feat_f = ocr_model(fake_imgs, return_features=True)
                _, ocr_feat_r = ocr_model(real_imgs.detach(), return_features=True)
                hard_ce_char = F.cross_entropy(
                    ocr_logits.view(-1, NUM_CLASSES), labels.view(-1),
                    reduction='none').view(B, CAPTCHA_LENGTH)
                if batch_idx % 100 == 0:
                    n_full = None
                elif batch_idx % 50 == 0:
                    n_full = 20
                elif batch_idx % 10 == 0:
                    n_full = 10
                else:
                    n_full = 0
                char_w, info = adaptive.compute_char_weights(
                    fake_imgs.detach(), labels, ocr_logits=ocr_logits.detach(),
                    max_samples=n_full, phase_c=True, len_apply=(False, False),
                    pair_sets=PAIR_SETS if args.pair_hard_weight > 0 else None,
                    pair_from=args.pair_from)
                soft = soft_ce(ocr_logits, labels, dist_full, eps_vec, 1.0)
                ocr_ce = (soft * char_w).sum() / float(B * CAPTCHA_LENGTH)
                ocr_feat = ocr_net.ocr_feature_loss(ocr_feat_f, ocr_feat_r)
                pair_mask = info.get("pair_mask")
                pair_ratio = float(pair_mask.float().mean()) if pair_mask is not None else 0.0
                pair_hard = torch.tensor(0.0, device=device)
                if (args.pair_hard_weight > 0 and pair_mask is not None
                        and bool(pair_mask.any())):
                    _pm = pair_mask.float()
                    pair_hard = (hard_ce_char * _pm).sum() / _pm.sum().clamp(min=1.0)
                ocr_loss = (args.lambda_soft * ocr_ce + args.lambda_feat * ocr_feat +
                            args.pair_hard_weight * pair_hard)
                ocr_w = float(char_w.mean())
                ocr_acc = (ocr_logits.detach().argmax(-1) == labels).all(dim=1).float().mean()

                g_loss = (adv_loss +
                          V28_LAMBDA_AUX_G * aux_loss +
                          V28_LAMBDA_FM * fm_loss +
                          V28_LAMBDA_CONTRAST * contrast_loss +
                          V28_LAMBDA_DIVERSITY * cur_div +
                          edge_lambda * cur_edge +
                          V28_LAMBDA_REALISM * cur_realism +
                          ocr_loss)

            if use_amp_flag:
                scaler.scale(g_loss).backward()
                scaler_unscale_(scaler, g_optim)
            else:
                g_loss.backward()
            _g_ok = all(p.grad is None or torch.isfinite(p.grad).all()
                        for p in G.parameters() if p.grad is not None)
            if not _g_ok:
                print(f"  [WARN] G grad NaN ep{epoch} step{global_step}, skip")
                g_optim.zero_grad()
                if use_amp_flag:
                    scaler.update()
                continue
            clip_gradients(G, args.clip_norm)
            if use_amp_flag:
                scaler.step(g_optim)
                scaler.update()
            else:
                g_optim.step()
            ema_g.update(G)
            global_step += 1

            d_gap = float(d_real.mean().detach() - d_fake.mean().detach())
            sums["d"] += float(d_loss.detach())
            sums["g"] += float(g_loss.detach())
            sums["aux"] += float(aux_loss.detach())
            sums["fm"] += float(fm_loss.detach())
            sums["ctr"] += float(contrast_loss.detach())
            sums["div"] += float(cur_div.detach())
            sums["edge"] += float(cur_edge.detach())
            sums["real"] += float(cur_realism.detach())
            sums["soft"] += float(ocr_ce.detach())
            sums["feat"] += float(ocr_feat.detach())
            sums["pair"] += float(pair_hard.detach())
            sums["pair_r"] += pair_ratio
            sums["w"] += ocr_w
            sums["acc"] += float(ocr_acc)
            sums["eg"] += edge_lambda
            n_batches += 1

            if global_step % 20 == 0:
                print(f"  [FT2] ep{epoch:3d} step{global_step:6d} "
                      f"D={float(d_loss):.3f} G={float(g_loss):.3f} "
                      f"soft={float(ocr_ce):.3f} pair={float(pair_hard):.4f}"
                      f"({pair_ratio:.1%}) feat={float(ocr_feat):.4f} "
                      f"w={ocr_w:.3f} acc={float(ocr_acc):.3f} dGap={d_gap:+.2f}")

            if args.max_steps and global_step >= args.max_steps:
                stop_after_epoch = True
                break

        n = max(n_batches, 1)
        avg = {k: v / n for k, v in sums.items()}
        ckpt_path = outdir / f"checkpoint_ft2_epoch_{epoch:03d}.pt"
        _need_save = (epoch % args.save_interval == 0 or epoch == 1
                      or (args.eval_interval > 0 and epoch % args.eval_interval == 0))
        if _need_save:
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "generator_state_dict": G.state_dict(),
                "discriminator_state_dict": D.state_dict(),
                "ema_state_dict": ema_g.state_dict(),
                "g_optimizer_state_dict": g_optim.state_dict(),
                "d_optimizer_state_dict": d_optim.state_dict(),
                "g_scheduler_state_dict": g_sched.state_dict(),
                "d_scheduler_state_dict": d_sched.state_dict(),
                "ada_p": float(ada.p),
                "best_mean_m9": best_mean_m9,
                "best_epoch": best_epoch,
                "base_ckpt": args.base_ckpt,
                "eps_info": eps_info,
                "ema_init": bool(args.ema_init),
                "bn_freeze": bool(args.bn_freeze),
            }, ckpt_path)
            print(f"  [SAVE] {ckpt_path.name}")
        # 保存 ckpt 后再采样, 避免 sample 更新 BN 污染 ckpt
        if args.sample_interval > 0 and (epoch % args.sample_interval == 0 or epoch == 1):
            save_sample(ema_g.ema_model, fixed_z, fixed_labels, epoch, global_step,
                        outdir, tag="ft2_ema")
            save_sample(G, fixed_z2, fixed_labels2, epoch, global_step,
                        outdir, tag="ft2_train")

        e_time = time.time() - e_start
        log_w.writerow([epoch, global_step, avg["d"], avg["g"], avg["aux"],
                        avg["fm"], avg["ctr"], avg["div"], avg["edge"],
                        avg["real"], avg["soft"], avg["feat"], avg["pair"],
                        avg["pair_r"], avg["w"], avg["acc"],
                        float(d_real.detach().mean()), float(d_fake.detach().mean()),
                        d_gap, avg["eg"], args.eps_mode, float(eps_vec.sum()),
                        round(e_time, 1)])
        print(f"  [FT2] —— Epoch {epoch:3d} D={avg['d']:.3f} G={avg['g']:.3f} "
              f"soft={avg['soft']:.3f} pair={avg['pair']:.4f}({avg['pair_r']:.1%}) "
              f"w={avg['w']:.3f} acc={avg['acc']:.3f} edge={avg['edge']:.4f} "
              f"dGap={float(d_real.detach().mean() - d_fake.detach().mean()):+.2f} "
              f"| {e_time:.0f}s")
        last_epoch = epoch

        # ─── 多 seed 评测 / guard / best ───────────────
        if args.eval_interval > 0 and args.eval_n > 0 and epoch % args.eval_interval == 0:
            results = []
            for seed in eval_seeds:
                tag = f"ft2_ep{epoch:03d}_s{seed}"
                eval_log = outdir / f"eval_{tag}.log"
                cmd = [sys.executable, str(ROOT / "evaluate_generated.py"),
                       "--ckpt", str(ckpt_path), "--n", str(args.eval_n),
                       "--tag", tag, "--output-dir", str(outdir),
                       "--seed", str(seed)]
                with open(eval_log, "w", encoding="utf-8") as ef:
                    subprocess.run(cmd, cwd=str(ROOT), stdout=ef,
                                   stderr=subprocess.STDOUT, check=False)
                m = parse_report(outdir / f"eval_report_{tag}.txt")
                if m:
                    results.append((seed, m))
                    print(f"  [EVAL] ep{epoch} s{seed} M1={m['m1_dddd']:.3f}/{m['m1_ppll']:.3f} "
                          f"M2={m['m2_dddd']}/{m['m2_ppll']} M6={m['m6']} "
                          f"M9={m['m9']:.4f} M8={m['m8']:.4f}")
                else:
                    print(f"  [EVAL] ep{epoch} s{seed} 报告缺失/解析失败")
            if results:
                mean = {k: _mean(results, k) for k in
                        ("m1_dddd", "m1_ppll", "m2_dddd", "m2_ppll", "m3_excess",
                         "m4_dddd", "m4_ppll", "m6", "m8", "m9")}
                m9s = [m["m9"] for _, m in results if m.get("m9") is not None]
                all_ok = len(results) == len(eval_seeds) and all(
                    _seed_ok(m, args) for _, m in results)
                details = {"epoch": epoch, "seeds": eval_seeds,
                           "mean": mean, "all_seed_ok": all_ok,
                           "per_seed": {str(seed): m for seed, m in results}}
                (outdir / f"eval_details_ft2_ep{epoch:03d}.json").write_text(
                    json.dumps(details, ensure_ascii=False, indent=1), encoding="utf-8")
                if all_ok and m9s and mean["m9"] is not None and mean["m9"] < best_mean_m9:
                    best_mean_m9 = mean["m9"]
                    best_epoch = epoch
                    shutil.copy2(ckpt_path, outdir / "checkpoint_ft2_best.pt")
                    (outdir / "best_metrics.json").write_text(
                        json.dumps(details, ensure_ascii=False, indent=1), encoding="utf-8")
                    print(f"  [BEST] ep{epoch} mean_M9={mean['m9']:.4f} "
                          f"M1={mean['m1_dddd']:.3f}/{mean['m1_ppll']:.3f} "
                          f"M2ppll={mean['m2_ppll']:.3f} M6={mean['m6']:.3f}")
                eval_w.writerow([epoch, mean["m1_dddd"], mean["m1_ppll"],
                                 mean["m2_dddd"], mean["m2_ppll"], mean["m3_excess"],
                                 mean["m4_dddd"], mean["m4_ppll"], mean["m6"],
                                 mean["m8"], mean["m9"],
                                 min(m9s) if m9s else None, max(m9s) if m9s else None,
                                 int(all_ok), best_mean_m9])
                if not all_ok:
                    bad = []
                    for seed, m in results:
                        if not _seed_ok(m, args):
                            bad.append(f"s{seed}")
                    print(f"  [GUARD] 多 seed 验收未过: {bad}; "
                          f"all_seed_ok={all_ok}")
                    if not args.no_stop_on_guard:
                        g_sched.step(); d_sched.step()
                        break

        g_sched.step()
        d_sched.step()
        if stop_after_epoch:
            print("  [FT2] max_steps reached, stop")
            break

    log_f.close()
    eval_f.close()

    # ─── 最终稳定别名: best (优先) 或 last -> checkpoint_v2_5ep.pt ───
    if args.final_name and args.epochs >= 5 and last_epoch >= 1:
        best_ckpt = outdir / "checkpoint_ft2_best.pt"
        last_ckpt = outdir / f"checkpoint_ft2_epoch_{last_epoch:03d}.pt"
        if best_ckpt.exists():
            src_ckpt, src_kind = best_ckpt, "best"
        elif last_ckpt.exists():
            src_ckpt, src_kind = last_ckpt, "last"
        else:
            src_ckpt, src_kind = None, None
        if src_ckpt is not None:
            final_ckpt = outdir / args.final_name
            shutil.copy2(src_ckpt, final_ckpt)
            metrics_src = (outdir / "best_metrics.json") if src_kind == "best" \
                else outdir / f"eval_details_ft2_ep{last_epoch:03d}.json"
            data = {}
            if metrics_src.exists():
                data = json.loads(metrics_src.read_text(encoding="utf-8"))
            data.update({
                "script": "finetune_v28_v2.py (5ep selected)",
                "base_ckpt": str(args.base_ckpt),
                "final_ckpt": str(final_ckpt),
                "source": src_kind,
                "epochs": args.epochs,
                "sched_epochs": sched_epochs,
                "eps_mode": args.eps_mode,
                "eval_seeds": eval_seeds,
                "selection": "all-seed guard; lowest mean M9",
            })
            metrics_path = outdir / "v2_5ep_metrics.json"
            metrics_path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
            print(f"  [FT2] final ckpt: {final_ckpt} ({src_kind})")
            print(f"  [FT2] metrics: {metrics_path}")

    print(f"  [FT2] done. best_epoch={best_epoch} best_mean_m9={best_mean_m9}")
    print(f"  [FT2] log: {log_path}")
    print(f"  [FT2] eval: {eval_path}")


if __name__ == "__main__":
    main()
