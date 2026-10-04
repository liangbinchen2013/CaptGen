# -*- coding: utf-8 -*-
"""
里程碑评测 — 生成图机器混淆画像 vs 真实画像 (M1-M8)

用法:
  python evaluate_generated.py --ckpt output_v28/checkpoint_epoch_400.pt
  python evaluate_generated.py --ckpt ... --n 2000 --tag ep300

流程:
  1. 加载 EMA 生成器, 按真实标签分布采样标签
  2. 生成 N 张 → ddddocr + ppllocr + CaptchaResNet 评估
  3. 输出与 ocr_error_stats_*.txt 同格式的 gen_error_stats_*.txt
  4. 对比 confusion_profile.json → M1-M8 报告 (console + report txt)

验收线 (以"匹配真实机器画像"为目标):
  M1 双OCR整图acc比 ∈ [0.85, 1.05]
  M2 字符错误率比 ∈ [0.8, 1.3]
  M4 <LEN>错误比 ∈ [0.5, 2.0] (与真实同量级, 既不硬切间隙也不过度粘连)
  M6 逐字符错误率MAE < 0.05
  M7 gap墨迹比贴近真实均值 ±0.15
  M8 CaptchaResNet整图acc ≥ 0.85
  M3 混淆分布JS散度仅作参考 (越低越接近真实画像)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (CHARS, NUM_CLASSES, CAPTCHA_LENGTH,
                    LATENT_DIM, GAP_STRIPS, GAP_INK_TAU, OUTPUT_DIR)

ROOT = Path(__file__).resolve().parent


# ─────────────────────────── 生成 ───────────────────────────
def load_generator(ckpt_path: Path, device):
    from models import Generator
    G = Generator().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # EMAGenerator.state_dict() = ema_model.state_dict() (无前缀), 直接可用
    state = ckpt.get("ema_state_dict") or ckpt.get("generator_state_dict")
    G.load_state_dict(state)
    G.eval()
    return G


def sample_labels(n: int, label_counts: dict):
    """按真实标签分布采样 (0缺失→不采, 1/5/O稀有→低频)"""
    chars = [c for c in CHARS if label_counts.get(c, 0) > 0]
    w = np.array([label_counts[c] for c in chars], dtype=np.float64)
    w /= w.sum()
    idx = np.random.choice(len(chars), size=(n, CAPTCHA_LENGTH), p=w)
    return torch.tensor([[CHARS.index(chars[j]) for j in row] for row in idx],
                        dtype=torch.long)


def generate(G, labels, device, batch=128):
    out = []
    with torch.no_grad():
        for i in range(0, labels.size(0), batch):
            z = torch.randn(labels[i:i+batch].size(0), LATENT_DIM, device=device)
            out.append(G(z, labels[i:i+batch].to(device)).cpu())
    return torch.cat(out)   # (N,1,H,W) tanh[-1,1]


# ─────────────────────────── 评估 ───────────────────────────
def eval_captcha_resnet(imgs, labels, device):
    """CaptchaResNet 整图 acc + 逐字符 (可读性代理 M8)"""
    import ocr_net
    model = ocr_net.load_ocr(device)
    accs, char_hits = [], 0
    with torch.no_grad():
        for i in range(0, imgs.size(0), 128):
            x = imgs[i:i+128].to(device)
            y = labels[i:i+128].to(device)
            pred = model(x).argmax(-1)
            accs.append((pred == y).all(dim=1).float().mean().item())
            char_hits += int((pred == y).sum())
    n_char = labels.numel()
    return float(np.mean(accs)), char_hits / n_char


def eval_third_party(imgs, labels):
    """ddddocr/ppllocr: 整图acc + (t,p)混淆对 + 长度/越界统计 (与 txt 同口径)"""
    import ddddocr
    try:
        from ppllocr import OCR as PPLLOCR
    except Exception:
        PPLLOCR = None

    # GPU 初始化失败自动回退 CPU (跨环境兼容)
    try:
        dddd = ddddocr.DdddOcr(show_ad=False, use_gpu=True)
    except Exception:
        dddd = ddddocr.DdddOcr(show_ad=False)
    ppll = PPLLOCR() if PPLLOCR else None

    results = {}
    for name, ocr in (("ddddocr", dddd), ("ppllocr", ppll)):
        if ocr is None:
            continue
        pair_counter = defaultdict(int)
        correct_img = total_img = total_err = 0
        len_err_total = oov_err_total = 0
        imgs_np = ((imgs * 0.5 + 0.5) * 255).byte().numpy()
        for i in range(imgs.size(0)):
            im = Image.fromarray(imgs_np[i, 0])
            buf = __import__("io").BytesIO()
            im.save(buf, format="JPEG")
            try:
                pred = (ocr.classification(buf.getvalue()) or "").strip().upper()
            except Exception:
                continue
            label = "".join(CHARS[int(t)] for t in labels[i].tolist())
            total_img += 1
            total_char = len(label)
            if pred == label:
                correct_img += 1
            if len(pred) == len(label):
                for t, p in zip(label, pred):
                    if t != p:
                        pair_counter[(t, p)] += 1
                        total_err += 1
                        if p not in CHARS:
                            oov_err_total += 1
            else:
                n = min(len(pred), len(label))
                for k in range(n):
                    t, p = label[k], pred[k]
                    if t != p:
                        pair_counter[(t, p)] += 1
                        total_err += 1
                        if p not in CHARS:
                            oov_err_total += 1
                pair_counter[("<LEN>", f"{len(label)}->{len(pred)}")] += 1
                total_err += 1
                len_err_total += 1
        results[name] = {
            "img_total": total_img,
            "img_correct": correct_img,
            "img_acc": correct_img / max(total_img, 1),
            "char_total": total_img * CAPTCHA_LENGTH,
            "char_err": total_err,
            "char_err_rate": total_err / max(total_img * CAPTCHA_LENGTH, 1),
            "len_err_total": len_err_total,
            "oov_err_total": oov_err_total,
            "pairs": dict(pair_counter),
        }
    return results


def gap_ink_metrics(imgs):
    """M7: gap墨迹比 (与 confusion_profile.real_gap 同口径)"""
    g = ((imgs + 1.0) * 0.5 * 255.0)
    B = g.size(0)
    bg = g.view(B, -1).median(dim=1).values.view(B, 1, 1, 1)
    ink = ((g - bg).abs() > GAP_INK_TAU * 255.0).float()
    prof = ink.mean(dim=(0, 1, 2))
    peak = float(prof.max().clamp(min=1e-6))
    strips = {f"gap{k+1}": float(prof[x0:x1].mean())
              for k, (x0, x1) in enumerate(GAP_STRIPS)}
    ratio = {k: v / peak for k, v in strips.items()}
    return {"peak_col_ink": peak, "strip_mean": strips, "ratio": ratio,
            "ratio_mean": float(np.mean(list(ratio.values())))}


# ─────────────────────────── 对比 ───────────────────────────
def js_divergence(p: dict, q: dict):
    """两个 (key→prob) 分布的 JS 散度 (log2, [0,1])"""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    def _cat(d):
        v = np.array([d.get(k, 0.0) for k in keys], dtype=np.float64)
        s = v.sum()
        return v / s if s > 0 else np.full(len(keys), 1.0 / len(keys))
    P, Q = _cat(p), _cat(q)
    M = 0.5 * (P + Q)

    def _kl(A, B):
        m = A > 0
        return float(np.sum(A[m] * np.log2(A[m] / B[m])))
    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)


def compare(gen_results, profile, gap):
    rep = []
    ok_all = True

    def check(name, value, lo=None, hi=None, fmt="%.4f"):
        nonlocal ok_all
        ok = ((lo is None or value >= lo) and (hi is None or value <= hi))
        ok_all &= ok
        rng = f"[{lo},{hi}]" if (lo is not None or hi is not None) else ""
        rep.append(f"  {'PASS' if ok else 'FAIL'} {name}: {fmt % value}  target{rng}")

    # M1/M2: 双OCR 整图acc比 & 字符错误率比
    for name in ("ddddocr", "ppllocr"):
        g, r = gen_results.get(name), profile["per_ocr"].get(name)
        if not g or not r:
            continue
        ratio_acc = g["img_acc"] / max(r["img_acc"], 1e-6)
        ratio_err = g["char_err_rate"] / max(r["char_err_rate"], 1e-6)
        check(f"M1 {name} acc比 fake/real", ratio_acc, 0.85, 1.05)
        check(f"M2 {name} 字符错误率比", ratio_err, 0.8, 1.3)

    # M3: 混淆分布 JS 散度 (双OCR合并, 去<LEN>) — 参考线, 不参与达标判定
    gen_pairs = defaultdict(float)
    for name, g in gen_results.items():
        for (t, p), c in g["pairs"].items():
            if t != "<LEN>":
                gen_pairs[(t, p)] += c
    real_pairs = defaultdict(float)
    for name in ("ddddocr", "ppllocr"):
        for k, c in profile["per_ocr"].get(name, {}).get("pairs", {}).items():
            t, p = k.split(">", 1)     # 预测字符可含 '>' (如 7->>), 仅切首段
            if t != "<LEN>":
                real_pairs[(t, p)] += c
    js = js_divergence(gen_pairs, real_pairs)
    rep.append(f"  INFO M3 混淆分布JS散度: {js:.4f}  (参考线<0.1, 越低越接近真实画像)")

    # M4: <LEN> 错误比 — 与真实数据同量级 (既避免硬切间隙, 也不过度粘连)
    for name in ("ddddocr", "ppllocr"):
        g, r = gen_results.get(name), profile["per_ocr"].get(name)
        if not g or not r or r.get("len_err_total", 0) == 0:
            continue
        ratio = (g["len_err_total"] / max(g["img_total"], 1)) / \
                (r["len_err_total"] / max(r["img_total"], 1))
        check(f"M4 {name} <LEN>错误比 (匹配真实)", ratio, 0.5, 2.0)

    # M5: 越界率
    for name in ("ddddocr", "ppllocr"):
        g, r = gen_results.get(name), profile["per_ocr"].get(name)
        if not g or not r:
            continue
        g_rate = g["oov_err_total"] / max(g["char_err"], 1)
        r_rate = r["oov_err_total"] / max(r["char_err"], 1)
        check(f"M5 {name} 越界率", g_rate, None, max(r_rate * 1.2, 0.02))

    # M7: gap 墨迹比 — 贴近真实画像 (既不硬切间隙, 也不过度粘连)
    real_gap_ratios = list(profile.get("real_gap", {}).get("ratio", {}).values())
    if real_gap_ratios:
        real_gap_mean = float(np.mean(real_gap_ratios))
        check("M7 gap墨迹比 (匹配真实)", gap["ratio_mean"],
              real_gap_mean - 0.15, real_gap_mean + 0.15)
        rep.append("       (真实画像 gap比: " + ", ".join(
            f"{k}={v:.2f}" for k, v in profile["real_gap"]["ratio"].items()) + ")")
    else:
        rep.append(f"  INFO M7 gap墨迹比: {gap['ratio_mean']:.4f} (缺少真实 gap 画像)")

    return rep, ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint 路径")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tag", default="eval", help="输出文件标签")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    profile = json.loads((ROOT / "confusion_profile.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)

    print(f"[1/4] 加载生成器 {args.ckpt} ...")
    G = load_generator(Path(args.ckpt), device)

    print(f"[2/4] 生成 {args.n} 张 (按真实标签分布采样) ...")
    labels = sample_labels(args.n, profile["label_counts"])
    imgs = generate(G, labels, device)

    print("[3/4] OCR 评估 (CaptchaResNet + ddddocr + ppllocr) ...")
    c_acc, c_char = eval_captcha_resnet(imgs, labels, device)
    print(f"  CaptchaResNet: 整图acc={c_acc:.4f} 逐字符acc={c_char:.4f}")
    gen_results = eval_third_party(imgs, labels)
    for name, g in gen_results.items():
        print(f"  {name}: 整图acc={g['img_acc']:.4f} 字符错误率={g['char_err_rate']:.4f} "
              f"<LEN>={g['len_err_total']} 越界={g['oov_err_total']}")
    gap = gap_ink_metrics(imgs)
    print(f"  gap墨迹比: {gap['ratio_mean']:.3f} " +
          " ".join(f"{k}={v:.2f}" for k, v in gap["ratio"].items()))

    # 写同格式画像文件 (输出目录随 OUTPUT_DIR)
    outdir = Path(OUTPUT_DIR)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    for name, g in gen_results.items():
        p = outdir / f"gen_error_stats_{name}_{args.tag}.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"OCR: {name}  [generated {args.tag}]\n")
            f.write(f"图片总数: {g['img_total']}\n")
            f.write(f"整图准确率: {g['img_acc']:.6f} ({g['img_correct']}/{g['img_total']})\n")
            f.write(f"字符错误数: {g['char_err']} / {g['char_total']}   "
                    f"字符错误率: {g['char_err_rate']:.6f}\n\n")
            f.write("真实字符->预测字符 次数 (占比)\n")
            tot = max(g["char_err"], 1)
            for (t, p_), c in sorted(g["pairs"].items(),
                                     key=lambda x: -x[1] if isinstance(x[1], int) else 0):
                f.write(f"{t}->{p_} {c} ({c/tot*100:.2f}%)\n")
        print(f"  [OK] {p.name}")

    print("[4/4] M1-M8 对比 ...")
    rep, ok_all = compare(gen_results, profile, gap)
    # M6: 逐字符错误率 MAE (生成侧 vs 真实侧, 同口径双OCR合并)
    gen_err_by_true = defaultdict(float)
    for name, g in gen_results.items():
        for (t, p_), c in g["pairs"].items():
            if t != "<LEN>":
                gen_err_by_true[t] += c
    # 采样标签中每字符出现次数 (×2 = 双OCR各评一遍)
    occ = defaultdict(int)
    for row in labels.tolist():
        for t in row:
            occ[CHARS[t]] += 1
    maes = []
    for ch in CHARS:
        denom = occ.get(ch, 0) * 2
        if denom == 0:
            continue
        g_rate = gen_err_by_true.get(ch, 0) / denom
        maes.append(abs(g_rate - profile["error_rate"].get(ch, 0.0)))
    m6 = float(np.mean(maes))
    ok6 = m6 < 0.05
    rep.append(f"  {'PASS' if ok6 else 'FAIL'} M6 逐字符错误率MAE: {m6:.4f}  target[<0.05]")

    # M8: 可读性
    ok8 = c_acc >= 0.85
    rep.append(f"  {'PASS' if ok8 else 'FAIL'} M8 CaptchaResNet整图acc: {c_acc:.4f}  target[>=0.85]")
    rep.append(f"       (人评待做: {outdir.name} 样图 @里程碑)")

    report = "\n".join([
        f"里程碑评测报告 [{args.tag}]",
        f"ckpt: {args.ckpt}   n={args.n}   seed={args.seed}",
        f"CaptchaResNet: 整图{c_acc:.4f} 逐字符{c_char:.4f}",
        *[f"{k}: acc={v['img_acc']:.4f} err_rate={v['char_err_rate']:.4f} "
          f"len_err={v['len_err_total']} oov={v['oov_err_total']}"
          for k, v in gen_results.items()],
        f"gap墨迹比: {gap['ratio_mean']:.3f}",
        "",
        "M1-M8:",
        *rep,
        "",
        "结论: " + ("全部达标" if ok_all and ok6 and ok8 else "部分未达标 (见 FAIL 项)"),
    ])
    rp = outdir / f"eval_report_{args.tag}.txt"
    rp.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[OK] 报告已保存: {rp}")


if __name__ == "__main__":
    main()
