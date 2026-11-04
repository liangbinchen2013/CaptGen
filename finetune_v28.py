# -*- coding: utf-8 -*-
"""
v28 短程低 LR 微调 — 只针对 M9 / M2 dddd，不动 v28 的 M1/M8/M4。

背景 (n=2000, v28 ep400):
  dddd 0.7955 (比 0.9965) / ppll 0.5965 (比 0.8789) —— M1 双 PASS;
  M2 dddd 0.853 (严格 0.9 口径略干净), M9 0.2301 FAIL;
  M3 超额 0.078 / M4 0.733/1.215 (匹配真实) / M6 0.0446 / M8 0.946 均 PASS.
v29-v31 从零重训都在 Phase A 掉了 basin (v31.1 A 末内部 0.905, 外部 0.609/0.379,
之后 280 轮平线). 因此这里不重训, 只从 v28 ep400 低 LR 微调 20-40 epoch:
  - 抑制头部超额混淆对 (v29 V29_SUPPRESS_PAIRS) + 行内重归一化;
  - 软 CE 独立权重 lambda_soft (默认 0.05, 不是外层 λ 0.15);
  - 特征匹配保留一个小权重 lambda_feat (默认 0.02);
  - 其余 D/G 损失、ADA、权重表保持 v28 行为
    (len_apply=(False,False) 即不启用 v32 的长度失败跳过逻辑);
  - 每 eval_interval 轮跑一次 evaluate_generated.py (n=2000), 按 M9 选 best,
    若 M1 dddd/ppll 跌破 guard (默认 0.85) 则停止.

用法:
  # 默认: output_v32/checkpoint_epoch_400.pt -> output_v32/checkpoint_ft_*
  python finetune_v28.py --epochs 30

  # 3 步冒烟 (小数据)
  python finetune_v28.py --data-dir <tiny> --output-dir <tmp> \
      --epochs 1 --batch-size 8 --max-steps 3 --eval-interval 0 --sample-interval 0

  # 关闭抑制对, 只做普通软 CE 对照
  python finetune_v28.py --no-suppress
"""
import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
import time
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
    GAP_INK_TAU, EDGE_LOSS_MIN, SEED,
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
    soft_ce, sample_real_labels, clip_gradients, _batch_quantile,
)

# v28 权重 (不要用 v32 config 里的 LAMBDA_AUX_G=1.0)
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

# v29 实证超额对 (v28 ep400 双 OCR, n>=5, ×>=2); 只抑制这些方向
SUPPRESS_PAIRS = (
    ("I", "H"), ("R", "T"), ("R", "X"), ("R", "F"), ("K", "T"), ("W", "T"),
    ("7", "T"), ("M", "T"), ("N", "H"), ("N", "A"), ("4", "A"), ("U", "B"),
    ("G", "C"), ("G", "E"), ("G", "O"), ("Q", "9"), ("I", "J"), ("I", "T"),
    ("V", "T"), ("L", "Z"), ("Z", "2"),
    ("I", "X"), ("R", "J"), ("R", "N"), ("R", "W"), ("R", "8"), ("T", "L"),
    ("A", "E"), ("T", "F"), ("Z", "R"), ("9", "T"), ("D", "P"), ("D", "J"),
    ("M", "N"), ("7", "1"), ("4", "9"),
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    manual_seed_all(seed)
    print(f"  [SEED] {seed}")


def build_targets(suppress=True):
    """v28 静态 ε + 可选抑制头部对后行内重归一化."""
    dist = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    for ch, row in CONFUSION_DIST.items():
        ci = CHARS.index(ch)
        for p, v in row.items():
            if p in CHARS:
                dist[ci, CHARS.index(p)] = v
    if suppress:
        for a, b in SUPPRESS_PAIRS:
            if a in CHARS and b in CHARS:
                dist[CHARS.index(a), CHARS.index(b)] = 0.0
        rs = dist.sum(dim=1, keepdim=True)
        dist = torch.where(rs > 0, dist / rs.clamp(min=1e-12), dist)
    eps = torch.zeros(NUM_CLASSES)
    for ch, v in OCR_ERROR_RATE.items():
        ci = CHARS.index(ch)
        if dist[ci].sum() > 0.5:
            eps[ci] = min(float(v), EPS_CAP)
    return dist, eps


def main():
    ap = argparse.ArgumentParser(description="v28 ep400 低 LR 微调 (M9/M2)")
    ap.add_argument("--base-ckpt", default=str(ROOT / "output_v32" / "checkpoint_epoch_400.pt"))
    ap.add_argument("--resume", default=None, help="从 finetune checkpoint 继续")
    ap.add_argument("--data-dir", default=str(ROOT / "训练数据"))
    ap.add_argument("--output-dir", default=str(ROOT / "output_v32"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5, help="G/D 微调学习率")
    ap.add_argument("--lr-min", type=float, default=5e-6)
    ap.add_argument("--d-updates", type=int, default=1)
    ap.add_argument("--clip-norm", type=float, default=CLIP_NORM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ada-p", type=float, default=1.0, help="v28 D 训练时 p 已饱和到 1")
    ap.add_argument("--lambda-soft", type=float, default=0.05)
    ap.add_argument("--lambda-feat", type=float, default=0.02)
    ap.add_argument("--no-suppress", action="store_true", help="不抑制头部超额对 (对照)")
    ap.add_argument("--eval-interval", type=int, default=5, help="0=不评测")
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--guard-m1", type=float, default=0.85,
                    help="M1 dddd/ppll 低于此值则停止 (0=关闭)")
    ap.add_argument("--ignore-guard", action="store_true", help="跌破 guard 也继续")
    ap.add_argument("--sample-interval", type=int, default=5, help="0=不采样")
    ap.add_argument("--save-interval", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 时只跑这么多 G step (冒烟)")
    args = ap.parse_args()
    args.d_updates = max(1, int(args.d_updates))

    seed_everything(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    print(f"  [FT] base={args.base_ckpt}")
    print(f"  [FT] out={outdir}")
    print(f"  [FT] lr={args.lr} epochs={args.epochs} soft={args.lambda_soft} "
          f"feat={args.lambda_feat} suppress={not args.no_suppress}")

    G = Generator().to(device)
    D = Discriminator().to(device)
    ema_g = EMAGenerator(G, decay=EMA_DECAY).to(device)
    g_optim = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    d_optim = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
    g_sched = optim.lr_scheduler.CosineAnnealingLR(g_optim, T_max=args.epochs,
                                                  eta_min=args.lr_min)
    d_sched = optim.lr_scheduler.CosineAnnealingLR(d_optim, T_max=args.epochs,
                                                  eta_min=args.lr_min)
    aux_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING_EPS)
    ada = ADAAugment(target=0.6, p_inc=2e-4, p_dec=5e-5, adjust_interval=4)
    ada.p = float(args.ada_p)
    scaler = get_scaler() if use_amp() else None
    use_amp_flag = scaler is not None

    start_epoch = 1
    global_step = 0
    best_m9 = float("inf")
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
        best_m9 = float(ckpt.get("best_m9", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", -1))
        print(f"  [FT] resume ep{start_epoch - 1}, global_step={global_step}")
    else:
        base = torch.load(args.base_ckpt, map_location=device, weights_only=True)
        G.load_state_dict(base["generator_state_dict"])
        D.load_state_dict(base["discriminator_state_dict"])
        ema_g.load_state_dict(base["ema_state_dict"])
        print(f"  [FT] loaded base epoch={base.get('epoch')} global_step={base.get('global_step')}")

    loader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    pin_memory = device_supports_pin_memory()
    fixed_z = torch.randn(16, LATENT_DIM, device=device)
    fixed_labels = sample_real_labels(16, device)
    fixed_z2 = torch.randn(16, LATENT_DIM, device=device)
    fixed_labels2 = sample_real_labels(16, device)

    adaptive = MultiOCRAdaptive(device=device, load_third_party=True)
    ocr_model = adaptive.ocr_models.get("captcha_resnet")
    if ocr_model is None:
        raise RuntimeError("CaptchaResNet 未加载, 无法微调")
    dist_supp, eps_supp = build_targets(suppress=not args.no_suppress)
    dist_supp = dist_supp.to(device)
    eps_supp = eps_supp.to(device)
    print(f"  [FT] suppress_pairs={0 if args.no_suppress else len(SUPPRESS_PAIRS)} "
          f"eps_sum={float(eps_supp.sum()):.2f}")

    log_path = outdir / "finetune_log.csv"
    log_new = not log_path.exists() or log_path.stat().st_size == 0
    log_f = open(log_path, "a", encoding="utf-8", buffering=1, newline="")
    log_w = csv.writer(log_f)
    if log_new:
        log_w.writerow(["epoch", "step", "d_loss", "g_loss", "aux", "fm",
                        "contrast", "div", "edge", "realism", "ocr_ce",
                        "ocr_feat", "ocr_w", "ocr_acc", "d_real", "d_fake",
                        "d_gap", "edge_lambda", "adv", "lambda_soft",
                        "lambda_feat", "epoch_time_s"])
    eval_path = outdir / "finetune_eval.csv"
    eval_new = not eval_path.exists() or eval_path.stat().st_size == 0
    eval_f = open(eval_path, "a", encoding="utf-8", buffering=1, newline="")
    eval_w = csv.writer(eval_f)
    if eval_new:
        eval_w.writerow(["epoch", "m1_dddd", "m1_ppll", "m2_dddd", "m2_ppll",
                         "m3_excess", "m4_dddd", "m4_ppll", "m6", "m8", "m9",
                         "valid", "best_m9"])

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

    t_start = time.time()
    stop_after_epoch = False
    for epoch in range(start_epoch, args.epochs + 1):
        e_start = time.time()
        sums = {k: 0.0 for k in ("d", "g", "aux", "fm", "ctr", "div", "edge",
                                 "real", "ce", "feat", "w", "acc", "eg")}
        n_batches = 0
        for batch_idx, (real_imgs, labels) in enumerate(loader):
            B = real_imgs.size(0)
            real_imgs = real_imgs.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            # ─── D ─────────────────────────────────────
            cached_real_features = None
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
                d_grad_norm = clip_gradients(D, args.clip_norm)
                if use_amp_flag:
                    scaler.step(d_optim)
                    scaler.update()
                else:
                    d_optim.step()
                ada.update_p(d_real.mean().detach().item())
                real_imgs.requires_grad_(False)
            else:
                d_grad_norm = torch.tensor(0.0, device=device)
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

                # OCR: 软目标 (抑制头部对) + 特征匹配; 不启用 v32 长度跳过逻辑
                ocr_logits, ocr_feat_f = ocr_model(fake_imgs, return_features=True)
                _, ocr_feat_r = ocr_model(real_imgs.detach(), return_features=True)
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
                    max_samples=n_full, phase_c=True, len_apply=(False, False))
                soft = soft_ce(ocr_logits, labels, dist_supp, eps_supp, 1.0)
                ocr_ce = (soft * char_w).sum() / float(B * CAPTCHA_LENGTH)
                ocr_feat = ocr_net.ocr_feature_loss(ocr_feat_f, ocr_feat_r)
                ocr_loss = args.lambda_soft * ocr_ce + args.lambda_feat * ocr_feat
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
            g_grad_norm = clip_gradients(G, args.clip_norm)
            if use_amp_flag:
                scaler.step(g_optim)
                scaler.update()
            else:
                g_optim.step()
            ema_g.update(G)
            global_step += 1

            d_gap = float(d_real.mean().detach() - d_fake.mean().detach())
            sums["d"] += float(d_loss.detach()); sums["g"] += float(g_loss.detach())
            sums["aux"] += float(aux_loss.detach()); sums["fm"] += float(fm_loss.detach())
            sums["ctr"] += float(contrast_loss.detach()); sums["div"] += float(cur_div.detach())
            sums["edge"] += float(cur_edge.detach()); sums["real"] += float(cur_realism.detach())
            sums["ce"] += float(ocr_ce.detach()); sums["feat"] += float(ocr_feat.detach())
            sums["w"] += ocr_w; sums["acc"] += float(ocr_acc); sums["eg"] += edge_lambda
            n_batches += 1

            if global_step % 20 == 0:
                print(f"  [FT] ep{epoch:3d} step{global_step:6d} "
                      f"D={float(d_loss):.3f} G={float(g_loss):.3f} "
                      f"soft={float(ocr_ce):.3f} feat={float(ocr_feat):.4f} "
                      f"w={ocr_w:.3f} acc={float(ocr_acc):.3f} "
                      f"edge={float(cur_edge):.4f} dGap={d_gap:+.2f}")

            if args.max_steps and global_step >= args.max_steps:
                stop_after_epoch = True
                break

        n = max(n_batches, 1)
        avg = {k: v / n for k, v in sums.items()}
        if args.sample_interval > 0 and (epoch % args.sample_interval == 0 or epoch == 1):
            save_sample(ema_g.ema_model, fixed_z, fixed_labels, epoch, global_step,
                        outdir, tag="ft_ema")
            save_sample(G, fixed_z2, fixed_labels2, epoch, global_step,
                        outdir, tag="ft_train")
        ckpt_path = outdir / f"checkpoint_ft_epoch_{epoch:03d}.pt"
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
                "best_m9": best_m9,
                "best_epoch": best_epoch,
                "base_ckpt": args.base_ckpt,
            }, ckpt_path)
            print(f"  [SAVE] {ckpt_path.name}")
        e_time = time.time() - e_start
        log_w.writerow([epoch, global_step, avg["d"], avg["g"], avg["aux"],
                        avg["fm"], avg["ctr"], avg["div"], avg["edge"],
                        avg["real"], avg["ce"], avg["feat"], avg["w"],
                        avg["acc"], float(d_real.detach().mean()),
                        float(d_fake.detach().mean()),
                        d_gap, avg["eg"], V28_ADV_LAMBDA, args.lambda_soft,
                        args.lambda_feat, round(e_time, 1)])
        print(f"  [FT] —— Epoch {epoch:3d} 完成 D={avg['d']:.3f} G={avg['g']:.3f} "
              f"soft={avg['ce']:.3f} feat={avg['feat']:.4f} w={avg['w']:.3f} "
              f"acc={avg['acc']:.3f} edge={avg['edge']:.4f} "
              f"dGap={float(d_real.detach().mean() - d_fake.detach().mean()):+.2f} | {e_time:.0f}s")

        if args.eval_interval > 0 and args.eval_n > 0 and epoch % args.eval_interval == 0:
            tag = f"ft_ep{epoch:03d}"
            eval_log = outdir / f"eval_{tag}.log"
            cmd = [sys.executable, str(ROOT / "evaluate_generated.py"),
                   "--ckpt", str(ckpt_path), "--n", str(args.eval_n),
                   "--tag", tag, "--output-dir", str(outdir),
                   "--seed", str(args.seed)]
            print(f"  [EVAL] {' '.join(cmd)}")
            with open(eval_log, "w", encoding="utf-8") as ef:
                subprocess.run(cmd, cwd=str(ROOT), stdout=ef,
                               stderr=subprocess.STDOUT, check=False)
            metrics = parse_report(outdir / f"eval_report_{tag}.txt")
            if metrics:
                m1d, m1p, m8, m9 = (metrics["m1_dddd"], metrics["m1_ppll"],
                                    metrics["m8"], metrics["m9"])
                valid = bool(m1d is not None and m1p is not None and m8 is not None
                             and m1d >= args.guard_m1 and m1p >= args.guard_m1
                             and m8 >= 0.85)
                if valid and m9 is not None and m9 < best_m9:
                    best_m9 = m9
                    best_epoch = epoch
                    shutil.copy2(ckpt_path, outdir / "checkpoint_ft_best.pt")
                    (outdir / "best_metrics.json").write_text(
                        json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
                    print(f"  [BEST] ep{epoch} M9={m9:.4f} M1={m1d:.3f}/{m1p:.3f}")
                eval_w.writerow([epoch, m1d, m1p, metrics["m2_dddd"], metrics["m2_ppll"],
                                 metrics["m3_excess"], metrics["m4_dddd"], metrics["m4_ppll"],
                                 metrics["m6"], m8, m9, int(valid), best_m9])
                print(f"  [EVAL] ep{epoch} M1={m1d:.3f}/{m1p:.3f} M2={metrics['m2_dddd']}/{metrics['m2_ppll']} "
                      f"M9={m9:.4f} M8={m8:.4f} valid={valid}")
                if args.guard_m1 > 0 and not args.ignore_guard and (
                        (m1d is not None and m1d < args.guard_m1) or
                        (m1p is not None and m1p < args.guard_m1)):
                    print(f"  [GUARD] M1 跌破 {args.guard_m1}，停止微调 (保留已保存 ckpt)")
                    g_sched.step(); d_sched.step()
                    break

        g_sched.step()
        d_sched.step()
        if stop_after_epoch:
            print("  [FT] max_steps reached, stop")
            break

    log_f.close()
    eval_f.close()
    print(f"  [FT] done. best_epoch={best_epoch} best_m9={best_m9}")
    print(f"  [FT] log: {log_path}")
    print(f"  [FT] eval: {eval_path}")


if __name__ == "__main__":
    main()
