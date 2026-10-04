"""
OCR准确率评估脚本 (带tqdm进度条)

评估4个OCR模型在训练数据上的准确率
用于自适应OCR监督方案的基准准确率
"""
import os
import sys
from pathlib import Path
from PIL import Image
import random
import time
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def get_image_paths(data_dir, sample_size=1000):
    """获取图片路径和标签"""
    print(f"\n加载图片路径...")
    all_paths = sorted(Path(data_dir).glob("batch_*/captcha_*.jpg"))
    print(f"总图片数: {len(all_paths)}")
    
    # 随机采样
    if sample_size and sample_size < len(all_paths):
        sampled = random.sample(all_paths, sample_size)
        print(f"采样数: {sample_size}")
    else:
        sampled = all_paths
    
    # 提取标签
    paths_labels = []
    for p in sampled:
        label = p.stem.replace('captcha_', '').upper()
        if len(label) == 4:
            paths_labels.append((str(p), label))
    
    print(f"有效样本数: {len(paths_labels)}")
    return paths_labels


def evaluate_ppllocr(paths_labels):
    """评估ppllocr准确率"""
    print("\n" + "="*50)
    print("评估 ppllocr")
    print("="*50)
    
    try:
        from ppllocr import OCR
        ocr = OCR()
        
        correct = 0
        total = 0
        
        for img_path, label in tqdm(paths_labels, desc="ppllocr"):
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            
            try:
                pred = ocr.classification(img_bytes).strip().upper()
                if len(pred) == 4 and pred == label:
                    correct += 1
                total += 1
            except:
                continue
        
        acc = correct / total if total > 0 else 0
        print(f"\nppllocr 准确率: {acc:.6f} ({correct}/{total})")
        return acc
        
    except Exception as e:
        print(f"ppllocr 评估失败: {e}")
        return None


def evaluate_ddddocr(paths_labels):
    """评估ddddocr准确率"""
    print("\n" + "="*50)
    print("评估 ddddocr")
    print("="*50)
    
    try:
        import ddddocr
        ocr = ddddocr.DdddOcr(show_ad=False,use_gpu=True)
        
        correct = 0
        total = 0
        
        for img_path, label in tqdm(paths_labels, desc="ddddocr"):
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            
            try:
                pred = ocr.classification(img_bytes).strip().upper()
                if len(pred) == 4 and pred == label:
                    correct += 1
                total += 1
            except:
                continue
        
        acc = correct / total if total > 0 else 0
        print(f"\nddddocr 准确率: {acc:.6f} ({correct}/{total})")
        return acc
        
    except Exception as e:
        print(f"ddddocr 评估失败: {e}")
        return None


def evaluate_paddleocr(paths_labels):
    """评估PaddleOCR准确率"""
    print("\n" + "="*50)
    print("评估 PaddleOCR")
    print("="*50)
    
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        correct = 0
        total = 0
        
        for img_path, label in tqdm(paths_labels, desc="PaddleOCR"):
            try:
                result = ocr.ocr(img_path, cls=True)
                if result and result[0]:
                    pred = ''.join([line[1][0] for line in result[0]]).strip().upper()
                    if len(pred) == 4 and pred == label:
                        correct += 1
                    total += 1
            except:
                continue
        
        acc = correct / total if total > 0 else 0
        print(f"\nPaddleOCR 准确率: {acc:.6f} ({correct}/{total})")
        return acc
        
    except Exception as e:
        print(f"PaddleOCR 评估失败: {e}")
        return None


def evaluate_easyocr(paths_labels):
    """评估EasyOCR准确率"""
    print("\n" + "="*50)
    print("评估 EasyOCR")
    print("="*50)
    
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)
        
        correct = 0
        total = 0
        
        for img_path, label in tqdm(paths_labels, desc="EasyOCR"):
            try:
                result = reader.readtext(img_path)
                if result:
                    pred = ''.join([text for _, text, _ in result]).strip().upper()
                    if len(pred) == 4 and pred == label:
                        correct += 1
                    total += 1
            except:
                continue
        
        acc = correct / total if total > 0 else 0
        print(f"\nEasyOCR 准确率: {acc:.6f} ({correct}/{total})")
        return acc
        
    except Exception as e:
        print(f"EasyOCR 评估失败: {e}")
        return None


def main():
    print("=" * 60)
    print("  OCR准确率评估 (自适应OCR监督基准)")
    print("=" * 60)
    
    # 设置随机种子
    random.seed(42)
    
    # 获取图片路径
    data_dir = Path("训练数据")
    paths_labels = get_image_paths(data_dir, sample_size=None)  # 全量评估
    
    # 评估每个OCR
    results = {}

    results['ppllocr'] = evaluate_ppllocr(paths_labels)
    results['ddddocr'] = evaluate_ddddocr(paths_labels)
    results['paddleocr'] = evaluate_paddleocr(paths_labels)
    results['easyocr'] = evaluate_easyocr(paths_labels)
    
    # 输出结果
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
    
    # 输出建议
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
        f.write("OCR基准准确率评估结果\n")
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
