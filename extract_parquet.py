"""
从 parquet 文件提取验证码图片，保存为 JPG
- 每 500 张放一个文件夹 batch_X
- 文件名 captcha_XXXX.jpg（保留原始大小写, 0-9, a-z, A-Z）
"""
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image


def extract_images():
    data_dir = Path(__file__).parent
    output_dir = data_dir / "训练数据"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 parquet 文件
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print("[ERROR] 未找到 .parquet 文件！")
        sys.exit(1)

    print(f"[INFO] 找到 {len(parquet_files)} 个 parquet 文件:")
    for f in parquet_files:
        fsize = f.stat().st_size / 1024 / 1024
        print(f"       {f.name} ({fsize:.1f} MB)")

    total_processed = 0
    batch_num = 1
    batch_count = 0

    for pf_idx, pf in enumerate(parquet_files):
        print(f"\n[INFO] 读取 {pf.name} ...")
        df = pd.read_parquet(pf)
        print(f"[INFO] {pf.name}: {len(df)} 行")

        for row_idx in range(len(df)):
            # --- 每 500 张新建一个 batch 目录 ---
            if batch_count >= 500:
                batch_num += 1
                batch_count = 0

            batch_dir = output_dir / f"batch_{batch_num}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            # --- 提取图片 ---
            img_data = df.iloc[row_idx]["image"]
            # img_data 是 (35,) 的 object 数组，每个元素是 (90,) 的 float32 数组
            # 重塑为 (35, 90)
            try:
                img_array = np.array([np.array(row).flatten() for row in img_data], dtype=np.float32)
            except Exception:
                # 如果直接转换不行，尝试另一种方式
                img_array = np.array(img_data.tolist(), dtype=np.float32).reshape(35, 90)

            # 值域 [0, 1] → [0, 255] uint8
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)

            # --- 提取标签 ---
            label_array = df.iloc[row_idx]["label"]
            # label_array 是 4 个 ASCII 码
            if hasattr(label_array, 'tolist'):
                label_codes = label_array.tolist()
            else:
                label_codes = list(label_array)
            label_str = "".join(chr(c) for c in label_codes)

            # --- 保存图片 ---
            filename = f"captcha_{label_str}.jpg"
            filepath = batch_dir / filename

            # 处理同名文件（极小概率，但加后缀防冲突）
            if filepath.exists():
                filepath = batch_dir / f"captcha_{label_str}_{batch_count:04d}.jpg"

            Image.fromarray(img_array).save(filepath, quality=95)

            total_processed += 1
            batch_count += 1

            # 进度显示
            if total_processed % 5000 == 0:
                print(f"[PROGRESS] 已处理 {total_processed} 张...")

        # 释放内存
        del df

    print(f"\n{'='*50}")
    print(f"[DONE] 提取完成！")
    print(f"[DONE] 共处理 {total_processed} 张图片")
    print(f"[DONE] 保存在 {output_dir.resolve()}")
    print(f"[DONE] 共 {batch_num} 个 batch 目录 (batch_1 ~ batch_{batch_num})")
    print(f"{'='*50}")


if __name__ == "__main__":
    extract_images()
