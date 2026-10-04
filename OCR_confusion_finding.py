# -*- coding: utf-8 -*-
"""
统计 ddddocr 和 ppllocr 在哪些字符上识别误差最多
全量测试训练数据目录下所有验证码

输出格式: 0->O 4145 (12.12%)
结果保存: ocr_error_stats_<engine>.txt (供 build_confusion_profile.py 使用)

直接运行: python OCR_confusion_finding.py
"""

import os
import re
import glob
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = str(Path(__file__).resolve().parent / "训练数据")
PATTERN = re.compile(r'^captcha_([0-9A-Za-z]{4})\.(?:jpg|jpeg|png|bmp)$', re.IGNORECASE)
TOP_N = 100


# ---------------------------------------------------------------- 数据收集
def batch_key(p):
    m = re.search(r'batch_(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else 0


def collect_samples(root):
    samples = []
    batch_dirs = sorted(glob.glob(os.path.join(root, 'batch_*')), key=batch_key)
    for d in batch_dirs:
        for fn in os.listdir(d):
            m = PATTERN.match(fn)
            if m:
                samples.append((os.path.join(d, fn), m.group(1).upper()))
    return samples


# ---------------------------------------------------------------- OCR 初始化
def get_ocr(engine):
    if engine == 'ddddocr':
        import ddddocr
        try:
            ocr = ddddocr.DdddOcr(show_ad=False, use_gpu=True)
        except Exception:
            ocr = ddddocr.DdddOcr(show_ad=False, use_gpu=False)
        return lambda b: ocr.classification(b)

    elif engine == 'ppllocr':
        from ppllocr import OCR
        ocr = OCR()
        return lambda b: ocr.classification(b)

    raise ValueError(f"未知引擎: {engine}")


# ---------------------------------------------------------------- 评估 + 统计
def evaluate(engine, samples, top=TOP_N, save_dir='.'):
    print("\n" + "=" * 60)
    print(f"评估 {engine}")
    print("=" * 60)

    try:
        classify = get_ocr(engine)
    except Exception as e:
        print(f"{engine} 初始化失败: {e}")
        return None

    pair_counter = Counter()   # (真实字符, 预测字符) -> 次数
    total_err = 0              # 字符级错误总数
    total_char = 0             # 参与比较的字符总数
    total_img = 0
    correct_img = 0
    failed_img = 0

    for path, label in tqdm(samples, desc=engine, ncols=100):
        try:
            with open(path, 'rb') as f:
                img_bytes = f.read()
            pred = (classify(img_bytes) or '').strip().upper()
        except Exception:
            failed_img += 1
            continue

        total_img += 1
        total_char += len(label)
        if pred == label:
            correct_img += 1

        if len(pred) == len(label):
            for t, p in zip(label, pred):
                if t != p:
                    pair_counter[(t, p)] += 1
                    total_err += 1
        else:
            n = min(len(pred), len(label))
            for i in range(n):
                t, p = label[i], pred[i]
                if t != p:
                    pair_counter[(t, p)] += 1
                    total_err += 1
            pair_counter[('<LEN>', f'{len(label)}->{len(pred)}')] += 1
            total_err += 1

    if total_img == 0:
        print(f"{engine}: 没有有效样本")
        return None

    acc = correct_img / total_img
    char_err_rate = total_err / total_char if total_char else 0

    print(f"\n图片总数: {total_img}   异常跳过: {failed_img}")
    print(f"整图准确率: {acc:.6f} ({correct_img}/{total_img})")
    print(f"字符错误数: {total_err} / {total_char}   字符错误率: {char_err_rate:.6f}")

    print("\n" + "-" * 60)
    print(f"字符错误分布 Top {top}   (真实字符 -> 预测字符)")
    print("-" * 60)
    for (t, p), c in pair_counter.most_common(top):
        pct = c / total_err * 100 if total_err else 0
        print(f"{t}->{p} {c} ({pct:.2f}%)")

    out = os.path.join(save_dir, f'ocr_error_stats_{engine}.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f"OCR: {engine}\n")
        f.write(f"图片总数: {total_img}   异常跳过: {failed_img}\n")
        f.write(f"整图准确率: {acc:.6f} ({correct_img}/{total_img})\n")
        f.write(f"字符错误数: {total_err} / {total_char}   字符错误率: {char_err_rate:.6f}\n\n")
        f.write("真实字符->预测字符 次数 (占比)\n")
        for (t, p), c in pair_counter.most_common():
            pct = c / total_err * 100 if total_err else 0
            f.write(f"{t}->{p} {c} ({pct:.2f}%)\n")
    print(f"\n完整结果已保存: {out}")

    return pair_counter


# ---------------------------------------------------------------- 主入口
def main():
    samples = collect_samples(ROOT)
    print(f"共收集 {len(samples)} 张验证码图片")
    if not samples:
        print("未找到样本, 请检查目录和文件名格式 (captcha_XXXX.jpg)")
        return

    for engine in ['ddddocr', 'ppllocr']:
        evaluate(engine, samples,
                 save_dir=str(Path(__file__).resolve().parent))


if __name__ == '__main__':
    main()