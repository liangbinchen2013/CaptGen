# -*- coding: utf-8 -*-
"""
错误画像分析 (v29 口径): 错误率 = 该错误次数 / 该真实字符的出现总数
- 真实侧分母: confusion_profile.json 的 label_counts (训练集全量扫描) × OCR数
- 生成侧分母: 生成标签中该字符出现次数 (与评测同 seed 复现) × OCR数
- 输出: 逐字符错误率、逐混淆对错误率、gen-vs-real 超额 (excess)、离谱清单
用法:
  python analyze_error_profile.py --gen-prefix output_v28/gen_error_stats --tag ep400 --n 2000 --seed 42
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARSET = set(CHARS)
PAIR_RE = re.compile(r'^(<LEN>|[0-9A-Z])->(\S*)\s+(\d+)\s+\(')


def parse_gen_txt(path: Path):
    pairs = {}
    img_total = correct = 0
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = PAIR_RE.match(ln)
        if m:
            pairs[(m.group(1), m.group(2))] = int(m.group(3))
            continue
        m = re.match(r'^图片总数:\s*(\d+)', ln)
        if m:
            img_total = int(m.group(1))
        m = re.match(r'^整图准确率:\s*[\d.]+\s+\((\d+)/', ln)
        if m:
            correct = int(m.group(1))
    return {"img_total": img_total, "img_correct": correct, "pairs": pairs}


def sample_labels(n, label_counts, seed):
    chars = [c for c in CHARS if label_counts.get(c, 0) > 0]
    w = np.array([label_counts[c] for c in chars], dtype=np.float64)
    w /= w.sum()
    np.random.seed(seed)
    idx = np.random.choice(len(chars), size=(n, 4), p=w)
    return [[chars[j] for j in row] for row in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-prefix", default="output_v28/gen_error_stats")
    ap.add_argument("--tag", default="ep400")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-rate", type=float, default=0.002,
                    help="离谱清单最小生成侧错误率")
    ap.add_argument("--excess-ratio", type=float, default=2.0,
                    help="离谱清单 gen/real 倍数门槛")
    args = ap.parse_args()

    profile = json.loads((ROOT / "confusion_profile.json").read_text(encoding="utf-8"))
    label_counts = profile["label_counts"]

    ocr_files = {}
    for name in ("ddddocr", "ppllocr"):
        p = ROOT / f"{args.gen_prefix}_{name}_{args.tag}.txt"
        if p.exists():
            ocr_files[name] = parse_gen_txt(p)
    if not ocr_files:
        raise SystemExit("未找到生成侧错误统计文件")

    # ── 生成侧字符分母 (复现评测采样) ──
    gen_labels = sample_labels(args.n, label_counts, args.seed)
    gen_occ = defaultdict(int)
    for row in gen_labels:
        for ch in row:
            gen_occ[ch] += 1
    n_ocr_gen = len(ocr_files)
    real_n_ocr = len(profile["per_ocr"])

    # ═══ 1. 逐字符错误率 (排除<LEN>, 含越界) ═══
    gen_err_by_true = defaultdict(int)
    gen_len_by_ocr = {}
    real_err_by_true = defaultdict(int)
    real_len_by_ocr = {}
    for name, g in ocr_files.items():
        len_err = 0
        oov = 0
        for (t, p), c in g["pairs"].items():
            if t == "<LEN>":
                len_err += c
                continue
            gen_err_by_true[t] += c
            if p not in CHARSET:
                oov += c
        g["len_err"] = len_err
        g["oov"] = oov
        gen_len_by_ocr[name] = len_err
    for name, r in profile["per_ocr"].items():
        real_len_by_ocr[name] = r.get("len_err_total", 0)
        for k, c in r.get("pairs", {}).items():
            t, p = k.split(">", 1)
            if t != "<LEN>":
                real_err_by_true[t] += c

    print("═" * 100)
    print(f"错误率口径: 错误数 / 该字符出现总数 (真实=训练集全量, 生成=采样{args.n}张×{n_ocr_gen}OCR)")
    print("═" * 100)
    print(f"{'ch':>3} {'真实总数':>8} {'真实错误':>7} {'真实率':>8} | {'生成总数':>7} {'生成错误':>7} {'生成率':>8} | {'超额':>8} {'倍数':>6}")
    rows = []
    for ch in CHARS:
        rt = label_counts.get(ch, 0) * real_n_ocr
        re_ = real_err_by_true.get(ch, 0)
        gt = gen_occ.get(ch, 0) * n_ocr_gen
        ge = gen_err_by_true.get(ch, 0)
        rr = re_ / rt if rt else 0.0
        gr = ge / gt if gt else 0.0
        ratio = (gr / rr) if rr > 1e-9 else float("inf") if gr > 0 else 0.0
        rows.append((ch, rt, re_, rr, gt, ge, gr, gr - rr, ratio))
    for ch, rt, re_, rr, gt, ge, gr, ex, ratio in rows:
        if gt == 0:
            continue
        flag = ""
        if gr >= args.min_rate and (ratio >= args.excess_ratio or rr == 0):
            flag = "  <== 离谱"
        print(f"{ch:>3} {rt:>8} {re_:>7} {rr*100:>7.2f}% | {gt:>7} {ge:>7} {gr*100:>7.2f}% | {ex*100:>+7.2f}pp {ratio:>6.2f}{flag}")

    # ═══ 2. 逐混淆对错误率 ═══
    print("\n" + "═" * 100)
    print("逐混淆对错误率 (分母=真实字符总数; 生成侧按各自字符出现数归一)")
    print("═" * 100)
    gen_pair = defaultdict(int)
    real_pair = defaultdict(int)
    for g in ocr_files.values():
        for (t, p), c in g["pairs"].items():
            if t != "<LEN>":
                gen_pair[(t, p)] += c
    for r in profile["per_ocr"].values():
        for k, c in r.get("pairs", {}).items():
            t, p = k.split(">", 1)
            if t != "<LEN>":
                real_pair[(t, p)] += c

    all_pairs = set(gen_pair) | set(real_pair)
    prs = []
    for (t, p) in all_pairs:
        rt = label_counts.get(t, 0) * real_n_ocr
        gt = gen_occ.get(t, 0) * n_ocr_gen
        rr = real_pair.get((t, p), 0) / rt if rt else 0.0
        gr = gen_pair.get((t, p), 0) / gt if gt else 0.0
        prs.append((t, p, real_pair.get((t, p), 0), rr, gen_pair.get((t, p), 0), gr, gr - rr))
    prs.sort(key=lambda x: -x[5])
    print(f"{'pair':>10} {'真实数':>6} {'真实率':>8} | {'生成数':>6} {'生成率':>8} | {'超额':>8} {'倍数':>6}")
    for t, p, rc, rr, gc, gr, ex in prs[:30]:
        ratio = gr / rr if rr > 1e-9 else float("inf")
        flag = "  <== 离谱" if (gr >= args.min_rate and ratio >= args.excess_ratio) else ""
        pstr = p if p else "(空/越界)"
        print(f"{t}->{pstr:>7} {rc:>6} {rr*100:>7.2f}% | {gc:>6} {gr*100:>7.2f}% | {ex*100:>+7.2f}pp {ratio:>6.2f}{flag}")

    # ═══ 3. 按需微调优先级: 字符级超额汇总 ═══
    print("\n" + "═" * 100)
    print("针对性微调优先级 (字符级: 生成侧错误贡献 & 与真实的超额)")
    print("═" * 100)
    focus = defaultdict(lambda: {"gen_err": 0, "real_err": 0, "excess": 0.0})
    for t, p, rc, rr, gc, gr, ex in prs:
        focus[t]["gen_err"] += gc
        focus[t]["real_err"] += rc
        focus[t]["excess"] += ex
    order = sorted(focus.items(), key=lambda kv: -kv[1]["excess"])
    print(f"{'ch':>3} {'生成错误':>8} {'真实错误':>8} {'超额总和':>10}")
    for ch, v in order[:15]:
        print(f"{ch:>3} {v['gen_err']:>8} {v['real_err']:>8} {v['excess']*100:>+9.2f}pp")

    # ═══ 4. <LEN> 长度错误 ═══
    print("\n" + "═" * 100)
    print("<LEN> 长度错误率 (反粘连关键)")
    print("═" * 100)
    for name, g in ocr_files.items():
        real_tot = profile["per_ocr"][name]["img_total"]
        real_len = real_len_by_ocr[name]
        rr = real_len / real_tot if real_tot else 0
        gr = g["len_err"] / g["img_total"] if g["img_total"] else 0
        print(f"  {name:>8}: 生成 {g['len_err']:>5}/{g['img_total']} = {gr*100:5.2f}% | "
              f"真实 {real_len:>7}/{real_tot} = {rr*100:5.2f}% | 比 {gr/rr if rr else 0:.3f}")

    # ═══ 5. 离谱清单 (pair级) ═══
    print("\n" + "═" * 100)
    print(f"离谱清单: 生成率>={args.min_rate*100:.1f}% 且 gen/real>={args.excess_ratio:.1f}x (或真实率=0)")
    print("═" * 100)
    n_bad = 0
    for t, p, rc, rr, gc, gr, ex in sorted(prs, key=lambda x: -x[5]):
        ratio = gr / rr if rr > 1e-9 else float("inf")
        if gr >= args.min_rate and (ratio >= args.excess_ratio or rr <= 1e-9):
            pstr = p if p else "(空/越界)"
            print(f"  {t}->{pstr:>7}: gen {gr*100:.2f}% (n={gc}) vs real {rr*100:.2f}% (n={rc})  ×{ratio if ratio!=float('inf') else 999:.1f}")
            n_bad += 1
    if n_bad == 0:
        print("  (无)")

    # 字符级 focus 建议
    print("\n[建议 focus 字符] (超额总和 top): " +
          ", ".join(ch for ch, _ in order[:10]))
    pair_focus = [f"{t}->{p}" for t, p, rc, rr, gc, gr, ex in
                  sorted(prs, key=lambda x: -(x[5] * max(x[4], 1)))[:10]
                  if gr >= args.min_rate * 0.5]
    print("[建议 focus 混淆对] " + ", ".join(pair_focus))


if __name__ == "__main__":
    main()
