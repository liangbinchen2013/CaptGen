"""
OCR准确率评估脚本 (全量版 + 中文路径修复)

评估2个OCR模型在训练数据上的准确率
用于自适应OCR监督方案的基准准确率
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import random
import time
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def read_image_safe(img_path):
    """
    安全读取图片，支持中文路径
    返回 numpy 数组 (BGR格式) 或 None
    """
    try:
        with open(img_path, 'rb') as f:
            data = f.read()
        img_array = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def get_image_paths(data_dir, sample_size=None):
    """获取全部图片路径和标签（无采样）"""
    print(f"\n加载图片路径...")
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"错误：数据目录不存在 -> {data_dir}")
        return []

    all_paths = sorted(data_dir.glob("batch_*/captcha_*.jpg"))
    total = len(all_paths)
    print(f"总图片数: {total}")

    # sample_size=None 表示全量，不做采样
    if sample_size is not None and sample_size < total:
        random.seed(42)
        sampled = random.sample(all_paths, sample_size)
        print(f"采样数: {sample_size}（注意：本脚本为全量模式，此参数被忽略）")
    else:
        sampled = all_paths
        print("全量模式：加载所有图片")

    # 提取标签（假设文件名格式 captcha_XXXX.jpg，XXXX为4位标签）
    paths_labels = []
    for p in sampled:
        label = p.stem.replace('captcha_', '').upper()
        if len(label) == 4 and label.isalnum():  # 只保留4位字母数字
            paths_labels.append((str(p), label))
        else:
            print(f"跳过无效文件名: {p.name}")

    print(f"有效样本数: {len(paths_labels)}")
    return paths_labels


def check_gpu():
    """检测GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU可用: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("GPU不可用，将使用CPU运行")
            return False
    except ImportError:
        print("PyTorch未安装，无法检测GPU")
        return False


def evaluate_paddleocr(paths_labels):
    """评估PaddleOCR准确率（使用图片数组，避免路径问题）"""
    print("\n" + "=" * 50)
    print("评估 PaddleOCR")
    print("=" * 50)

    try:
        from paddleocr import PaddleOCR
        # 启用GPU（如果可用），关闭日志
        ocr = PaddleOCR(
            use_textline_orientation=False,
            lang='en',
            device="cpu"
        )



        correct = 0
        total = 0

        for img_path, label in tqdm(paths_labels, desc="PaddleOCR"):
            try:
                img = read_image_safe(img_path)
                if img is None:
                    continue

                result = ocr.ocr(img, cls=True)
                if result and result[0]:
                    pred = ''.join([line[1][0] for line in result[0]]).strip().upper()
                    if len(pred) == 4 and pred == label:
                        correct += 1
                    total += 1
            except Exception:
                continue

        acc = correct / total if total > 0 else 0
        print(f"\nPaddleOCR 准确率: {acc:.6f} ({correct}/{total})")
        return acc

    except Exception as e:
        print(f"PaddleOCR 评估失败: {e}")
        return None


def evaluate_easyocr(paths_labels):
    """评估EasyOCR准确率（使用图片数组，避免路径问题）"""
    print("\n" + "=" * 50)
    print("评估 EasyOCR")
    print("=" * 50)

    try:
        import easyocr
        gpu_available = check_gpu()
        reader = easyocr.Reader(['en'], gpu=gpu_available)

        correct = 0
        total = 0

        for img_path, label in tqdm(paths_labels, desc="EasyOCR"):
            try:
                img = read_image_safe(img_path)
                if img is None:
                    continue

                result = reader.readtext(img)
                if result:
                    pred = ''.join([text for _, text, _ in result]).strip().upper()
                    if len(pred) == 4 and pred == label:
                        correct += 1
                    total += 1
            except Exception:
                continue

        acc = correct / total if total > 0 else 0
        print(f"\nEasyOCR 准确率: {acc:.6f} ({correct}/{total})")
        return acc

    except Exception as e:
        print(f"EasyOCR 评估失败: {e}")
        return None


def main():
    print("=" * 60)
    print("  OCR准确率评估 (全量评估)")
    print("=" * 60)

    random.seed(42)

    # ============ 配置 ============
    data_dir = Path("训练数据")     # 数据目录（支持中文）
    sample_size = None            # None 表示全量，不采样
    # ==============================

    paths_labels = get_image_paths(data_dir, sample_size=sample_size)
    if not paths_labels:
        print("没有有效图片，程序退出")
        return

    # 评估
    results = {}
    #results['easyocr'] = evaluate_easyocr(paths_labels)
    results['paddleocr'] = evaluate_paddleocr(paths_labels)

    # 汇总
    print("\n" + "=" * 60)
    print("  评估结果汇总")
    print("=" * 60)

    valid_results = {}
    for name, acc in results.items():
        if acc is not None:
            print(f"  {name}: {acc:.4f} ({acc*100:.1f}%)")
            valid_results[name] = acc
        else:
            print(f"  {name}: 评估失败")

    if len(valid_results) >= 2:
        print("\n" + "=" * 60)
        print("  自适应OCR监督建议")
        print("=" * 60)
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
        print("\n推荐使用的OCR模型:")
        for name, acc in sorted_results[:3]:
            threshold = acc + 0.05
            print(f"  {name}:")
            print(f"    基准准确率: {acc*100:.1f}%")
            print(f"    欺骗阈值:  {threshold*100:.1f}% (基准+5%)")

    # 保存结果
    output_file = Path("ocr_baseline_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("OCR基准准确率评估结果（全量）\n")
        f.write("=" * 40 + "\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本数: {len(paths_labels)}\n\n")
        for name, acc in results.items():
            if acc is not None:
                f.write(f"{name}: {acc:.4f}\n")
            else:
                f.write(f"{name}: 评估失败\n")

    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()