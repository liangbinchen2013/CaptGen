# -*- coding: utf-8 -*-
"""
解析 ocr_error_stats_ddddocr.txt / ocr_error_stats_ppllocr.txt
生成 confusion_profile.json —— 机器混淆画像资产

输出:
  label_counts[c]       每字符在训练集中的出现次数
  error_rate[c]         每字符真实错误率 (含越界预测, 不含<LEN>, 双OCR平均)
  confusion_dist[c][p]  有向混淆分布 (仅36类内, 按行归一化, 双OCR合并)
  confusable_pairs      实证混淆对 (双向证据充分)
  per_ocr.*             双OCR原始聚合 (供 evaluate_generated.py 做 M1-M6 对比)
  real_gap              真实数据 gap 墨迹画像 (抽样, 供 M7 对比)

用法: python build_confusion_profile.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARSET = set(CHARS)
TXTS = {
    "ddddocr": ROOT / "ocr_error_stats_ddddocr.txt",
    "ppllocr": ROOT / "ocr_error_stats_ppllocr.txt",
}
OUT = ROOT / "confusion_profile.json"

# 形如:  I->L 640 (1.18%)   /   <LEN>->4->5 1546 (2.85%)   /   Z->  1 (0.00%)
PAIR_RE = re.compile(r'^(<LEN>|[0-9A-Z])->(\S*)\s+(\d+)\s+\(')


def parse_txt(path: Path):
    meta = {"img_total": 0, "img_skip": 0, "img_correct": 0,
            "char_total": 0, "char_err": 0}
    pairs = defaultdict(int)
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = PAIR_RE.match(ln)
        if m:
            pairs[(m.group(1), m.group(2))] += int(m.group(3))
            continue
        m = re.match(r'^图片总数:\s*(\d+)\s+异常跳过:\s*(\d+)', ln)
        if m:
            meta["img_total"], meta["img_skip"] = int(m.group(1)), int(m.group(2))
            continue
        m = re.match(r'^整图准确率:\s*([\d.]+)\s+\((\d+)/(\d+)\)', ln)
        if m:
            meta["img_acc"] = float(m.group(1))
            meta["img_correct"] = int(m.group(2))
            continue
        m = re.match(r'^字符错误数:\s*(\d+)\s*/\s*(\d+)\s+字符错误率:\s*([\d.]+)', ln)
        if m:
            meta["char_err"], meta["char_total"] = int(m.group(1)), int(m.group(2))
            meta["char_err_rate"] = float(m.group(3))
    return meta, dict(pairs)


def scan_label_counts():
    """从文件名统计每字符出现次数 (error_rate 分母)"""
    cnt = defaultdict(int)
    pat = re.compile(r'^captcha_([0-9A-Za-z]{4})\.(?:jpg|jpeg|png|bmp)$', re.IGNORECASE)
    n = 0
    for d in sorted(ROOT.glob("训练数据/batch_*")):
        for f in d.iterdir():
            m = pat.match(f.name)
            if m:
                for ch in m.group(1).upper():
                    cnt[ch] += 1
                n += 1
    return dict(cnt), n


def scan_real_gap(sample_n=2000):
    """抽样真实数据: 列墨迹剖面 → gap/峰值 比 (与 train.py gap 损失同定义)"""
    import numpy as np
    from PIL import Image
    files = []
    for d in sorted(ROOT.glob("训练数据/batch_*")):
        files += [f for f in d.glob("captcha_*.jpg")]
        if len(files) >= sample_n:
            break
    files = files[:sample_n]
    prof = np.zeros(90, dtype=np.float64)
    used = 0
    for f in files:
        try:
            im = np.asarray(Image.open(f).convert("L"), dtype=np.float64)
        except Exception:
            continue
        if im.shape != (35, 90):
            continue
        bg = np.median(im)
        prof += (np.abs(im - bg) > 40).astype(np.float64).mean(axis=0)
        used += 1
    if used == 0:
        return {}
    prof /= used
    peak = float(prof.max())
    strips = {"gap1": (21, 25), "gap2": (43, 47), "gap3": (66, 70)}
    out = {"sample_n": used, "peak_col_ink": peak, "strip_mean": {},
           "ratio": {}}
    for name, (x0, x1) in strips.items():
        s = float(prof[x0:x1].mean())
        out["strip_mean"][name] = s
        out["ratio"][name] = s / peak if peak > 0 else 0.0
    return out


def main():
    ocrs = {}
    for name, path in TXTS.items():
        if not path.exists():
            print(f"[ERROR] 缺少 {path.name}, 请先运行 OCR_confusion_finding.py")
            sys.exit(1)
        meta, pairs = parse_txt(path)
        ocrs[name] = {"meta": meta, "pairs": pairs}
        print(f"[OK] {name}: img_acc={meta.get('img_acc'):.4f} "
              f"char_err_rate={meta.get('char_err_rate'):.4f} pairs={len(pairs)}")

    label_counts, n_img = scan_label_counts()
    print(f"[OK] 标签扫描: {n_img} 图, 字符总数 {sum(label_counts.values())}")

    # ── 按真实字符聚合错误 (含越界, 不含<LEN>) ──
    err_by_true = defaultdict(float)          # 双OCR合并
    in_vocab = defaultdict(lambda: defaultdict(float))
    len_err = {n: {} for n in ocrs}
    oov_err = defaultdict(float)
    for name, data in ocrs.items():
        for (t, p), c in data["pairs"].items():
            if t == "<LEN>":
                len_err[name][p] = len_err[name].get(p, 0) + c
                continue
            err_by_true[t] += c
            if p in CHARSET:
                in_vocab[t][p] += c
            else:
                oov_err[name] += c

    error_rate = {}
    for ch in CHARS:
        denom = label_counts.get(ch, 0) * len(ocrs)   # 双OCR各看一遍
        error_rate[ch] = round(err_by_true.get(ch, 0) / denom, 6) if denom else 0.0

    confusion_dist = {}
    for ch in CHARS:
        row = in_vocab.get(ch, {})
        s = sum(row.values())
        if s > 0:
            confusion_dist[ch] = {p: round(c / s, 6)
                                  for p, c in sorted(row.items(), key=lambda x: -x[1])}

    # ── 实证混淆对 ──
    pair_tot = defaultdict(int)   # (a,b) 有向合计
    pair_ocr = defaultdict(lambda: defaultdict(int))
    for name, data in ocrs.items():
        for (t, p), c in data["pairs"].items():
            if t == "<LEN>" or p not in CHARSET:
                continue
            if t == p:
                continue
            pair_tot[(t, p)] += c
            pair_ocr[(t, p)][name] += c

    def include(a, b):
        fwd, rev = pair_tot.get((a, b), 0), pair_tot.get((b, a), 0)
        comb = fwd + rev
        if comb < 500:
            return False
        # 双向强混淆: 两方向都>=100
        if min(fwd, rev) >= 100:
            return True
        # 单向强混淆: 主方向双OCR各>=150 (两模型独立共识)
        dom = (a, b) if fwd >= rev else (b, a)
        return all(pair_ocr[dom][n] >= 150 for n in ocrs)

    seen, pairs_out = set(), []
    for (a, b), c in sorted(pair_tot.items(), key=lambda x: -x[1]):
        key = tuple(sorted((a, b)))
        if key in seen or a == b:
            continue
        if include(a, b):
            seen.add(key)
            pairs_out.append([a, b])

    print(f"[OK] 实证混淆对: {len(pairs_out)} 对")
    for a, b in pairs_out:
        print(f"     {a}<->{b}: fwd={pair_tot.get((a,b),0)} rev={pair_tot.get((b,a),0)}")

    real_gap = scan_real_gap()
    if real_gap:
        print(f"[OK] 真实 gap 画像: " + " ".join(
            f"{k}={v:.2f}" for k, v in real_gap["ratio"].items()))

    profile = {
        "version": "1.0",
        "source": [p.name for p in TXTS.values()],
        "charset": CHARS,
        "label_counts": label_counts,
        "label_total": sum(label_counts.values()),
        "per_ocr": {
            name: {
                **data["meta"],
                "len_err": len_err[name],
                "len_err_total": sum(len_err[name].values()),
                "oov_err_total": oov_err[name],
                "pairs": {f"{t}>{p}": c for (t, p), c in data["pairs"].items()},
            } for name, data in ocrs.items()
        },
        "error_rate": error_rate,
        "confusion_dist": confusion_dist,
        "confusable_pairs": pairs_out,
        "real_gap": real_gap,
    }
    OUT.write_text(json.dumps(profile, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n[OK] 已写入 {OUT.name}")


if __name__ == "__main__":
    main()
