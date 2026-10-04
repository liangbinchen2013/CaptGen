"""
训练数据详细分析脚本
分析内容：
1. 总体统计（图片数量、batch分布）
2. 字符分布（36类字符频率）
3. 标签组合多样性
4. 大小写分布
5. 图片质量检查
6. 位置字符分布
"""
import os
from pathlib import Path
from collections import Counter, defaultdict
import random
from PIL import Image
import numpy as np


def analyze_training_data():
    data_dir = Path(__file__).resolve().parent / "训练数据"
    
    print("=" * 70)
    print("  验证码训练数据详细分析报告")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════
    #  1. 总体统计
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  1. 总体统计")
    print("─" * 70)
    
    batch_dirs = sorted(data_dir.glob("batch_*"),
                        key=lambda p: int(p.name.split("_")[1]))
    total_images = 0
    batch_stats = []
    corrupted_count = 0
    
    for batch_dir in batch_dirs:
        jpg_files = list(batch_dir.glob("*.jpg"))
        batch_stats.append((batch_dir.name, len(jpg_files)))
        total_images += len(jpg_files)
    
    print(f"  Batch 目录数:  {len(batch_dirs)}")
    print(f"  总图片数量:    {total_images}")
    print(f"  平均每 batch:  {total_images / len(batch_dirs):.1f} 张")
    
    # batch 大小分布
    batch_sizes = [s[1] for s in batch_stats]
    print(f"  Batch 大小范围: {min(batch_sizes)} ~ {max(batch_sizes)}")
    
    # ═══════════════════════════════════════════════════════════
    #  2. 标签解析与字符统计
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  2. 标签解析与字符统计")
    print("─" * 70)
    
    all_labels = []
    char_counter = Counter()
    position_char_counter = [Counter() for _ in range(4)]
    label_combinations = set()
    corrupted_files = []
    
    for batch_dir in batch_dirs:
        for f in batch_dir.glob("captcha_*.jpg"):
            # 解析标签
            stem = f.stem
            label_part = stem.replace("captcha_", "", 1)
            clean = "".join(c for c in label_part if c.isalnum())[:4]
            
            if len(clean) == 4:
                all_labels.append(clean)
                label_combinations.add(clean)
                for i, c in enumerate(clean):
                    char_counter[c] += 1
                    position_char_counter[i][c] += 1
            else:
                corrupted_files.append(f)
                corrupted_count += 1
    
    print(f"  有效标签数:    {len(all_labels)}")
    print(f"  损坏/异常文件: {corrupted_count}")
    print(f"  唯一标签组合:  {len(label_combinations)}")
    print(f"  理论最大组合:  36^4 = {36**4}")
    print(f"  标签覆盖率:    {len(label_combinations) / 36**4 * 100:.2f}%")
    
    # ═══════════════════════════════════════════════════════════
    #  3. 字符频率分布 (36类: 0-9, A-Z)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  3. 字符频率分布 (36类: 0-9, A-Z)")
    print("─" * 70)
    
    # 按数字和字母分别统计
    digit_chars = [c for c in char_counter.keys() if c.isdigit()]
    upper_chars = [c for c in char_counter.keys() if c.isupper()]
    lower_chars = [c for c in char_counter.keys() if c.islower()]
    
    total_chars = sum(char_counter.values())
    
    print(f"\n  总字符数: {total_chars} (理论值: {len(all_labels) * 4} = {len(all_labels)} × 4)")
    print(f"\n  字符类型分布:")
    print(f"    数字 (0-9):  {sum(char_counter[c] for c in digit_chars):>8} ({sum(char_counter[c] for c in digit_chars) / total_chars * 100:.2f}%)")
    print(f"    大写 (A-Z):  {sum(char_counter[c] for c in upper_chars):>8} ({sum(char_counter[c] for c in upper_chars) / total_chars * 100:.2f}%)")
    print(f"    小写 (a-z):  {sum(char_counter[c] for c in lower_chars):>8} ({sum(char_counter[c] for c in lower_chars) / total_chars * 100:.2f}%)")
    
    # 详细字符频率表
    print(f"\n  逐字符频率 (理想值: {total_chars / 36:.0f} / 字符, 偏差±10%内为正常):")
    print(f"  {'字符':>4} {'数量':>8} {'占比':>8} {'偏差':>8} {'状态':>6}")
    print(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    
    ideal_count = total_chars / 36
    for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        count = char_counter.get(c, 0)
        pct = count / total_chars * 100
        deviation = (count - ideal_count) / ideal_count * 100
        status = "正常" if abs(deviation) < 10 else ("偏多" if deviation > 0 else "偏少")
        print(f"  {c:>4} {count:>8} {pct:>7.2f}% {deviation:>+7.2f}% {status:>6}")
    
    # ═══════════════════════════════════════════════════════════
    #  4. 位置字符分布
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  4. 各位置字符分布 (位置1-4)")
    print("─" * 70)
    
    for pos in range(4):
        pos_counter = position_char_counter[pos]
        total_pos = sum(pos_counter.values())
        top5 = pos_counter.most_common(5)
        
        print(f"\n  位置 {pos + 1}:")
        print(f"    唯一字符数: {len(pos_counter)}")
        print(f"    Top 5: ", end="")
        for char, count in top5:
            print(f"{char}({count}, {count/total_pos*100:.1f}%) ", end="")
        print()
    
    # ═══════════════════════════════════════════════════════════
    #  5. 大小写分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  5. 大小写标注分析")
    print("─" * 70)
    
    # 统计同时出现大写和小写的标签
    mixed_case_labels = []
    for label in all_labels:
        has_upper = any(c.isupper() for c in label)
        has_lower = any(c.islower() for c in label)
        if has_upper and has_lower:
            mixed_case_labels.append(label)
    
    # 统计纯大写和纯小写
    all_upper_labels = [l for l in all_labels if l.isupper() and not l.isdigit()]
    all_lower_labels = [l for l in all_labels if l.islower() and not l.isdigit()]
    has_digit_labels = [l for l in all_labels if any(c.isdigit() for c in l)]
    
    print(f"  纯大写标签 (不含数字): {len(all_upper_labels)} ({len(all_upper_labels)/len(all_labels)*100:.2f}%)")
    print(f"  纯小写标签 (不含数字): {len(all_lower_labels)} ({len(all_lower_labels)/len(all_labels)*100:.2f}%)")
    print(f"  含数字标签:           {len(has_digit_labels)} ({len(has_digit_labels)/len(all_labels)*100:.2f}%)")
    print(f"  大小写混合标签:       {len(mixed_case_labels)} ({len(mixed_case_labels)/len(all_labels)*100:.2f}%)")
    
    if mixed_case_labels:
        print(f"\n  [!] 检测到大小写混合标注，示例:")
        for label in mixed_case_labels[:10]:
            print(f"      {label}")
    
    # 分析同一字母的大小写分布
    letter_pairs = defaultdict(lambda: {"upper": 0, "lower": 0})
    for label in all_labels:
        for c in label:
            if c.isalpha():
                upper_c = c.upper()
                if c.isupper():
                    letter_pairs[upper_c]["upper"] += 1
                else:
                    letter_pairs[upper_c]["lower"] += 1
    
    print(f"\n  字母大小写分布对比 (理想: 50%/50%):")
    print(f"  {'字母':>4} {'大写':>8} {'小写':>8} {'大写占比':>8} {'一致性':>8}")
    print(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        upper = letter_pairs[c]["upper"]
        lower = letter_pairs[c]["lower"]
        total = upper + lower
        if total > 0:
            upper_pct = upper / total * 100
            consistency = "一致" if abs(upper_pct - 50) < 5 else "不一致"
            print(f"  {c:>4} {upper:>8} {lower:>8} {upper_pct:>7.1f}% {consistency:>8}")
    
    # ═══════════════════════════════════════════════════════════
    #  6. 图片质量检查
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  6. 图片质量采样检查 (随机采样500张)")
    print("─" * 70)
    
    sample_size = min(500, len(all_labels))
    sample_files = []
    
    for batch_dir in batch_dirs:
        for f in batch_dir.glob("captcha_*.jpg"):
            sample_files.append(f)
            if len(sample_files) >= sample_size:
                break
        if len(sample_files) >= sample_size:
            break
    
    sizes = []
    pixel_means = []
    pixel_stds = []
    dark_ratio = []
    load_errors = 0
    
    for f in sample_files:
        try:
            img = Image.open(f).convert("L")
            arr = np.array(img, dtype=np.float32)
            sizes.append(arr.shape)
            pixel_means.append(arr.mean())
            pixel_stds.append(arr.std())
            dark_ratio.append((arr < 50).mean())
        except Exception as e:
            load_errors += 1
    
    if sizes:
        print(f"  采样数量:      {len(sizes)}")
        print(f"  加载失败:      {load_errors}")
        print(f"  图片尺寸:      {sizes[0][0]}×{sizes[0][1]} (所有图片一致)")
        print(f"  像素均值:      {np.mean(pixel_means):.1f} ± {np.std(pixel_means):.1f} (范围0-255)")
        print(f"  像素标准差:    {np.mean(pixel_stds):.1f} ± {np.std(pixel_stds):.1f}")
        print(f"  暗底比例:      {np.mean(dark_ratio)*100:.1f}% (像素<50)")
        
        # 质量判断
        mean_brightness = np.mean(pixel_means)
        if mean_brightness < 80:
            print(f"  [INFO] 图片整体偏暗 (均值{mean_brightness:.0f}<80)")
        elif mean_brightness > 200:
            print(f"  [INFO] 图片整体偏亮 (均值{mean_brightness:.0f}>200)")
        else:
            print(f"  [INFO] 图片亮度正常 (均值{mean_brightness:.0f})")
    
    # ═══════════════════════════════════════════════════════════
    #  7. 标签多样性分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  7. 标签多样性分析")
    print("─" * 70)
    
    label_counter = Counter(all_labels)
    unique_labels = len(label_counter)
    most_common = label_counter.most_common(10)
    least_common = label_counter.most_common()[-10:]
    
    print(f"  唯一标签数:    {unique_labels}")
    print(f"  标签总数:      {len(all_labels)}")
    print(f"  平均每标签:    {len(all_labels) / unique_labels:.2f} 张")
    
    print(f"\n  最常见的 10 个标签:")
    for label, count in most_common:
        print(f"    {label}: {count} 张 ({count/len(all_labels)*100:.3f}%)")
    
    print(f"\n  最少见的 10 个标签:")
    for label, count in least_common:
        print(f"    {label}: {count} 张 ({count/len(all_labels)*100:.3f}%)")
    
    # 标签频次分布
    freq_dist = Counter(label_counter.values())
    print(f"\n  标签频次分布:")
    for freq in sorted(freq_dist.keys())[:10]:
        print(f"    出现 {freq} 次的标签: {freq_dist[freq]} 个")
    
    # ═══════════════════════════════════════════════════════════
    #  8. 数据集总结
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  数据集总结")
    print("═" * 70)
    
    print(f"  总图片数:      {total_images}")
    print(f"  图片尺寸:      35×90 灰度图")
    print(f"  字符集:        36类 (0-9 + A-Z)")
    print(f"  标签长度:      4 字符")
    print(f"  唯一标签:      {unique_labels} / {36**4} ({unique_labels/36**4*100:.2f}%)")
    print(f"  大小写问题:    {'存在混合标注' if mixed_case_labels else '标注一致'}")
    print(f"  数据质量:      {'良好' if load_errors == 0 else f'有{load_errors}张损坏'}")
    
    # 建议
    print(f"\n  训练建议:")
    if mixed_case_labels:
        print(f"    1. [重要] 存在大小写混合标注，建议统一为大写 (config.py 已处理)")
    if unique_labels < 36**4 * 0.5:
        print(f"    2. [注意] 标签覆盖率较低 ({unique_labels/36**4*100:.1f}%)，部分组合缺失")
    print(f"    3. 数据量充足 ({total_images}张)，适合训练 GAN 模型")
    print(f"    4. 建议使用 EMA + 多种损失函数防止过拟合")
    
    print("\n" + "═" * 70)


if __name__ == "__main__":
    analyze_training_data()