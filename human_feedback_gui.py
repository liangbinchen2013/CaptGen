"""
GAN人工反馈收集工具
功能：
1. 加载G/D模型，生成图片
2. 获取D的打分和loss
3. 人工输入评语
4. 保存到JSON，支持断点续传
"""
import json
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime

from config import *
from models import Generator, Discriminator
from device_utils import DEVICE


class GANFeedbackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GAN人工反馈收集工具")
        self.root.geometry("900x800")
        
        # 数据文件路径
        self.root_dir = Path(__file__).resolve().parent
        self.data_file = self.root_dir / "human_feedback.json"
        self.feedback_data = self.load_data()
        
        # 加载模型
        self.load_models()
        
        # 统计数据
        self.stats_data = {}
        
        # 创建GUI
        self.create_widgets()
        
        # 开始计算统计数据（后台线程）
        import threading
        self.stats_thread = threading.Thread(target=self.compute_stats, daemon=True)
        self.stats_thread.start()
        
        # 生成第一张图片
        self.generate_image()
    
    def load_models(self):
        """加载 G 和 D 模型 (自动选择 OUTPUT_DIR 中最新的 checkpoint)"""
        print(f"正在加载模型到 {DEVICE}...")

        ckpts = sorted((self.root_dir / OUTPUT_DIR).glob("checkpoint_epoch_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"未找到 checkpoint: {self.root_dir / OUTPUT_DIR}/checkpoint_epoch_*.pt")
        ckpt_path = ckpts[-1]
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

        # 加载 Generator (优先 EMA 权重)
        self.G = Generator().to(DEVICE)
        self.G.load_state_dict(ckpt.get("ema_state_dict", ckpt["generator_state_dict"]))
        self.G.eval()

        # 加载 Discriminator
        self.D = Discriminator().to(DEVICE)
        self.D.load_state_dict(ckpt["discriminator_state_dict"])
        self.D.eval()

        print(f"模型加载完成! ({ckpt_path.name})")
    
    def load_data(self):
        """加载历史数据，支持断点续传"""
        if self.data_file.exists():
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"已加载 {len(data)} 条历史反馈")
                return data
        return []
    
    def save_data(self):
        """保存数据到JSON"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
    
    def create_widgets(self):
        """创建GUI组件"""
        # 顶部框架 - 图片和D打分
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 左侧 - 图片
        img_frame = ttk.LabelFrame(top_frame, text="生成图片")
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.img_label = ttk.Label(img_frame)
        self.img_label.pack(padx=5, pady=5)
        
        # 右侧 - D打分信息
        score_frame = ttk.LabelFrame(top_frame, text="判别器打分")
        score_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.d_real_label = ttk.Label(score_frame, text="D(真实): --", font=("Arial", 12))
        self.d_real_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.d_fake_label = ttk.Label(score_frame, text="D(生成): --", font=("Arial", 12))
        self.d_fake_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.d_gap_label = ttk.Label(score_frame, text="D Gap: --", font=("Arial", 12))
        self.d_gap_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.d_loss_label = ttk.Label(score_frame, text="D Loss: --", font=("Arial", 12))
        self.d_loss_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.aux_acc_label = ttk.Label(score_frame, text="Aux准确率: --", font=("Arial", 12))
        self.aux_acc_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # 标签信息
        self.label_info = ttk.Label(score_frame, text="标签: --", font=("Arial", 12))
        self.label_info.pack(anchor=tk.W, padx=10, pady=2)
        
        # 统计数据区域
        stats_frame = ttk.LabelFrame(self.root, text="500张样本统计 (计算中...)")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=8, font=("Courier", 10))
        self.stats_text.pack(fill=tk.X, padx=5, pady=5)
        self.stats_text.insert(tk.END, "正在计算统计数据...")
        self.stats_text.configure(state='disabled')
        
        # 中间框架 - 人工评语
        feedback_frame = ttk.LabelFrame(self.root, text="人工评语")
        feedback_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 预设评语按钮
        preset_frame = ttk.Frame(feedback_frame)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        presets = [
            ("清晰可读", "clear"),
            ("模糊不清", "blurry"),
            ("字符错误", "wrong_char"),
            ("噪声太多", "noisy"),
            ("对比度低", "low_contrast"),
            ("边缘粗糙", "rough_edge"),
            ("样式单一", "uniform"),
            ("整体良好", "good"),
        ]
        
        for text, tag in presets:
            btn = ttk.Button(preset_frame, text=text, 
                           command=lambda t=tag: self.add_preset_feedback(t))
            btn.pack(side=tk.LEFT, padx=2)
        
        # 自定义评语输入
        custom_frame = ttk.Frame(feedback_frame)
        custom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(custom_frame, text="自定义评语:").pack(side=tk.LEFT)
        self.custom_entry = ttk.Entry(custom_frame, width=50)
        self.custom_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(custom_frame, text="添加", 
                   command=self.add_custom_feedback).pack(side=tk.LEFT)
        
        # 评语列表
        self.feedback_list = scrolledtext.ScrolledText(feedback_frame, height=4)
        self.feedback_list.pack(fill=tk.X, padx=5, pady=5)
        
        # 底部框架 - 操作按钮
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(bottom_frame, text="生成新图片", 
                   command=self.generate_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="保存反馈", 
                   command=self.save_feedback).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="跳过此图", 
                   command=self.skip_image).pack(side=tk.LEFT, padx=5)
        
        # 统计信息
        self.stats_label = ttk.Label(bottom_frame, 
                                     text=f"已收集: {len(self.feedback_data)} 条")
        self.stats_label.pack(side=tk.RIGHT, padx=10)
        
        # 当前图片的临时评语
        self.current_feedback = []
        self.current_data = {}
    
    def generate_image(self):
        """生成新图片并获取D打分"""
        # 随机生成标签
        self.current_labels = torch.randint(0, NUM_CLASSES, (1, CAPTCHA_LENGTH), device=DEVICE)
        
        # 生成图片
        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        with torch.no_grad():
            self.current_fake = self.G(z, self.current_labels)
        
        # 加载一张真实图片进行对比
        real_img, real_label = self.get_random_real_image()
        
        # 获取D打分 (batch_size=2避免MinibatchStdDev的NaN)
        # 拼成batch_size=2
        fake_for_d = torch.cat([self.current_fake, self.current_fake], dim=0)
        real_for_d = torch.cat([real_img, real_img], dim=0)
        label_for_d = torch.cat([self.current_labels, real_label], dim=0)
        with torch.no_grad():
            d_fake_out, aux_fake_out, _ = self.D(fake_for_d, label_for_d)
            d_real_out, aux_real_out, _ = self.D(real_for_d, label_for_d)
        
        # 取第一个样本的结果
        d_fake = d_fake_out[0:1]
        d_real = d_real_out[0:1]
        aux_fake = aux_fake_out[0:1]
        aux_real = aux_real_out[0:1]
        
        # 计算D Loss (Hinge Loss)
        d_loss_fake = torch.nn.functional.relu(1.0 + d_fake).mean()
        d_loss_real = torch.nn.functional.relu(1.0 - d_real).mean()
        d_loss = (d_loss_fake + d_loss_real) / 2
        
        # 辅助损失
        aux_loss = torch.nn.functional.cross_entropy(
            aux_fake.view(-1, NUM_CLASSES), 
            self.current_labels.view(-1)
        )
        
        # 辅助准确率 - 生成图片
        aux_preds = aux_fake.argmax(dim=-1)
        aux_acc = (aux_preds == self.current_labels).all(dim=1).float().item()
        
        # 辅助准确率 - 真实图片
        aux_real_preds = aux_real.argmax(dim=-1)
        aux_acc_real = (aux_real_preds == real_label).all(dim=1).float().item()
        
        # 保存当前数据
        self.current_data = {
            "d_fake": d_fake.item(),
            "d_real": d_real.item(),
            "d_gap": (d_real.item() - d_fake.item()),
            "d_loss": d_loss.item(),
            "d_loss_fake": d_loss_fake.item(),
            "d_loss_real": d_loss_real.item(),
            "aux_loss": aux_loss.item(),
            "aux_acc_fake": aux_acc,
            "aux_acc_real": aux_acc_real,
            "labels": self.current_labels.cpu().numpy().tolist()[0],
            "label_str": ''.join([CHARS[l] for l in self.current_labels.cpu().numpy()[0]]),
        }
        
        # 显示图片
        self.display_image(self.current_fake)
        
        # 更新D打分显示
        self.update_score_display()
        
        # 清空评语
        self.current_feedback = []
        self.feedback_list.delete(1.0, tk.END)
    
    def get_random_real_image(self):
        """从训练数据中随机获取一张真实图片"""
        import random
        from PIL import Image as PILImage
        
        # 获取所有batch目录
        data_dir = self.root_dir / "训练数据"
        batch_dirs = sorted(list(data_dir.glob("batch_*")))
        
        # 随机选择一个batch
        batch_dir = random.choice(batch_dirs)
        
        # 获取该batch中的所有图片
        img_files = sorted(list(batch_dir.glob("captcha_*.jpg")))
        
        # 随机选择一张图片
        img_file = random.choice(img_files)
        
        # 加载图片
        img = PILImage.open(img_file).convert('L')
        img_np = np.array(img, dtype=np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # 解析标签
        label_str = img_file.stem.replace('captcha_', '').upper()
        label_idx = [CHARS.index(c) for c in label_str[:4]]
        label_tensor = torch.tensor([label_idx], device=DEVICE)
        
        return img_tensor, label_tensor
    
    def compute_stats(self):
        """计算500张样本的统计数据"""
        import random
        from PIL import Image as PILImage
        
        print("开始计算统计数据 (500张样本)...")
        
        # 收集所有图片路径
        data_dir = self.root_dir / "训练数据"
        all_images = []
        for batch_dir in data_dir.glob("batch_*"):
            for img_file in batch_dir.glob("captcha_*.jpg"):
                all_images.append(img_file)
        
        # 随机选择500张
        sample_size = min(500, len(all_images))
        sampled_images = random.sample(all_images, sample_size)
        
        # 收集统计数据
        d_fake_list = []
        d_real_list = []
        d_gap_list = []
        d_loss_list = []
        aux_loss_list = []
        aux_acc_fake_list = []
        aux_acc_real_list = []
        
        for i, img_file in enumerate(sampled_images):
            if i % 50 == 0:
                print(f"计算进度: {i}/{sample_size}")
            
            try:
                # 加载真实图片
                img = PILImage.open(img_file).convert('L')
                img_np = np.array(img, dtype=np.float32) / 127.5 - 1.0
                real_img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                # 解析标签
                label_str = img_file.stem.replace('captcha_', '').upper()
                label_idx = [CHARS.index(c) for c in label_str[:4]]
                real_label = torch.tensor([label_idx], device=DEVICE)
                
                # 生成图片（使用相同标签）
                z = torch.randn(1, LATENT_DIM, device=DEVICE)
                with torch.no_grad():
                    fake_img = self.G(z, real_label)
                
                # 检查是否生成了NaN
                if torch.isnan(fake_img).any():
                    print(f"警告: 第{i}张生成图片包含NaN，跳过")
                    continue
                
                # 获取D打分 (batch_size=2避免MinibatchStdDev的NaN)
                with torch.no_grad():
                    # 拼成batch_size=2
                    fake_for_d = torch.cat([fake_img, fake_img], dim=0)
                    real_for_d = torch.cat([real_img, real_img], dim=0)
                    label_for_d = torch.cat([real_label, real_label], dim=0)
                    
                    d_fake_out, aux_fake_out, _ = self.D(fake_for_d, label_for_d)
                    d_real_out, aux_real_out, _ = self.D(real_for_d, label_for_d)
                
                # 取第一个样本
                d_fake = d_fake_out[0:1]
                d_real = d_real_out[0:1]
                aux_fake = aux_fake_out[0:1]
                aux_real = aux_real_out[0:1]
                
                # 检查D输出是否为NaN
                if torch.isnan(d_fake).any() or torch.isnan(d_real).any():
                    print(f"警告: 第{i}张D输出包含NaN，跳过")
                    continue
                
                # 计算指标
                d_loss_fake = torch.nn.functional.relu(1.0 + d_fake).mean()
                d_loss_real = torch.nn.functional.relu(1.0 - d_real).mean()
                d_loss = (d_loss_fake + d_loss_real) / 2
                
                aux_loss = torch.nn.functional.cross_entropy(
                    aux_fake.view(-1, NUM_CLASSES), 
                    real_label.view(-1)
                )
                
                aux_preds_fake = aux_fake.argmax(dim=-1)
                aux_acc_fake = (aux_preds_fake == real_label).all(dim=1).float().item()
                
                aux_preds_real = aux_real.argmax(dim=-1)
                aux_acc_real = (aux_preds_real == real_label).all(dim=1).float().item()
                
                # 收集数据
                d_fake_list.append(d_fake.item())
                d_real_list.append(d_real.item())
                d_gap_list.append(d_real.item() - d_fake.item())
                d_loss_list.append(d_loss.item())
                aux_loss_list.append(aux_loss.item())
                aux_acc_fake_list.append(aux_acc_fake)
                aux_acc_real_list.append(aux_acc_real)
                
            except Exception as e:
                print(f"处理第{i}张图片时出错: {e}")
                continue
        
        # 检查是否有有效数据
        if len(d_fake_list) == 0:
            print("错误: 没有有效的统计数据!")
            self.stats_data = {"error": "没有有效数据"}
            self.root.after(0, self.update_stats_display)
            return
        
        print(f"成功计算了 {len(d_fake_list)} 张图片的统计数据")
        
        # 计算统计
        self.stats_data = {
            "sample_size": len(d_fake_list),
            "d_fake": {
                "min": min(d_fake_list),
                "max": max(d_fake_list),
                "avg": sum(d_fake_list) / len(d_fake_list),
            },
            "d_real": {
                "min": min(d_real_list),
                "max": max(d_real_list),
                "avg": sum(d_real_list) / len(d_real_list),
            },
            "d_gap": {
                "min": min(d_gap_list),
                "max": max(d_gap_list),
                "avg": sum(d_gap_list) / len(d_gap_list),
            },
            "d_loss": {
                "min": min(d_loss_list),
                "max": max(d_loss_list),
                "avg": sum(d_loss_list) / len(d_loss_list),
            },
            "aux_loss": {
                "min": min(aux_loss_list),
                "max": max(aux_loss_list),
                "avg": sum(aux_loss_list) / len(aux_loss_list),
            },
            "aux_acc_fake": {
                "min": min(aux_acc_fake_list),
                "max": max(aux_acc_fake_list),
                "avg": sum(aux_acc_fake_list) / len(aux_acc_fake_list),
            },
            "aux_acc_real": {
                "min": min(aux_acc_real_list),
                "max": max(aux_acc_real_list),
                "avg": sum(aux_acc_real_list) / len(aux_acc_real_list),
            },
        }
        
        # 打印一些示例数据
        print(f"\n示例数据 (前5个):")
        print(f"  d_fake: {d_fake_list[:5]}")
        print(f"  d_real: {d_real_list[:5]}")
        print(f"  d_gap: {d_gap_list[:5]}")
        
        # 更新GUI显示
        self.root.after(0, self.update_stats_display)
        print("统计数据计算完成!")
    
    def update_stats_display(self):
        """更新统计数据的显示"""
        self.stats_text.configure(state='normal')
        self.stats_text.delete(1.0, tk.END)
        
        # 检查是否有错误
        if "error" in self.stats_data:
            self.stats_text.insert(tk.END, f"错误: {self.stats_data['error']}")
            self.stats_text.configure(state='disabled')
            return
        
        s = self.stats_data
        text = f"样本数: {s['sample_size']}\n"
        text += "=" * 60 + "\n"
        text += f"{'指标':<15} {'Min':>10} {'Max':>10} {'Avg':>10}\n"
        text += "-" * 60 + "\n"
        text += f"{'D(真实)':<15} {s['d_real']['min']:>10.4f} {s['d_real']['max']:>10.4f} {s['d_real']['avg']:>10.4f}\n"
        text += f"{'D(生成)':<15} {s['d_fake']['min']:>10.4f} {s['d_fake']['max']:>10.4f} {s['d_fake']['avg']:>10.4f}\n"
        text += f"{'D Gap':<15} {s['d_gap']['min']:>10.4f} {s['d_gap']['max']:>10.4f} {s['d_gap']['avg']:>10.4f}\n"
        text += f"{'D Loss':<15} {s['d_loss']['min']:>10.4f} {s['d_loss']['max']:>10.4f} {s['d_loss']['avg']:>10.4f}\n"
        text += f"{'Aux Loss':<15} {s['aux_loss']['min']:>10.4f} {s['aux_loss']['max']:>10.4f} {s['aux_loss']['avg']:>10.4f}\n"
        text += f"{'Aux准确率(假)':<15} {s['aux_acc_fake']['min']:>10.4f} {s['aux_acc_fake']['max']:>10.4f} {s['aux_acc_fake']['avg']:>10.4f}\n"
        text += f"{'Aux准确率(真)':<15} {s['aux_acc_real']['min']:>10.4f} {s['aux_acc_real']['max']:>10.4f} {s['aux_acc_real']['avg']:>10.4f}\n"
        text += "=" * 60 + "\n"
        
        # 添加分析
        text += "\n分析:\n"
        if s['d_gap']['avg'] < 1.0:
            text += "⚠️ D Gap过小，判别器可能太弱\n"
        elif s['d_gap']['avg'] > 3.0:
            text += "⚠️ D Gap过大，判别器可能太强\n"
        else:
            text += "✅ D Gap正常\n"
        
        if s['aux_acc_fake']['avg'] > 0.95:
            text += "⚠️ Aux准确率(假)过高，可能存在欺骗\n"
        else:
            text += "✅ Aux准确率(假)正常\n"
        
        self.stats_text.insert(tk.END, text)
        self.stats_text.configure(state='disabled')
    
    def display_image(self, tensor_img):
        """显示图片"""
        # 转换tensor为PIL Image
        img_np = tensor_img.cpu().squeeze().numpy()
        img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # 放大显示
        img_pil = img_pil.resize((270, 105), Image.NEAREST)
        
        # 转换为Tkinter可用格式
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # 更新标签
        self.img_label.configure(image=img_tk)
        self.img_label.image = img_tk  # 保持引用
    
    def update_score_display(self):
        """更新D打分显示"""
        self.d_real_label.configure(text=f"D(真实): {self.current_data['d_real']:.4f}")
        self.d_fake_label.configure(text=f"D(生成): {self.current_data['d_fake']:.4f}")
        self.d_gap_label.configure(text=f"D Gap: {self.current_data['d_gap']:.4f}")
        self.d_loss_label.configure(text=f"D Loss: {self.current_data['d_loss']:.4f}")
        self.aux_acc_label.configure(text=f"Aux准确率(生成): {self.current_data['aux_acc_fake']:.4f} | (真实): {self.current_data['aux_acc_real']:.4f}")
        self.label_info.configure(text=f"标签: {self.current_data['label_str']}")
    
    def add_preset_feedback(self, tag):
        """添加预设评语"""
        feedback_text = {
            "clear": "清晰可读",
            "blurry": "模糊不清",
            "wrong_char": "字符错误",
            "noisy": "噪声太多",
            "low_contrast": "对比度低",
            "rough_edge": "边缘粗糙",
            "uniform": "样式单一",
            "good": "整体良好",
        }
        
        text = feedback_text.get(tag, tag)
        self.current_feedback.append({"tag": tag, "text": text})
        self.feedback_list.insert(tk.END, f"[预设] {text}\n")
    
    def add_custom_feedback(self):
        """添加自定义评语"""
        text = self.custom_entry.get().strip()
        if text:
            self.current_feedback.append({"tag": "custom", "text": text})
            self.feedback_list.insert(tk.END, f"[自定义] {text}\n")
            self.custom_entry.delete(0, tk.END)
    
    def save_feedback(self):
        """保存当前图片的反馈"""
        if not self.current_feedback:
            print("请先添加评语!")
            return
        
        # 构建反馈数据
        feedback_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "labels": self.current_data["labels"],
            "label_str": self.current_data["label_str"],
            "d_fake": self.current_data["d_fake"],
            "d_real": self.current_data["d_real"],
            "d_gap": self.current_data["d_gap"],
            "d_loss": self.current_data["d_loss"],
            "d_loss_fake": self.current_data["d_loss_fake"],
            "d_loss_real": self.current_data["d_loss_real"],
            "aux_loss": self.current_data["aux_loss"],
            "aux_acc_fake": self.current_data["aux_acc_fake"],
            "aux_acc_real": self.current_data["aux_acc_real"],
            "human_feedback": self.current_feedback,
        }
        
        # 添加到数据列表
        self.feedback_data.append(feedback_entry)
        
        # 保存到文件
        self.save_data()
        
        # 更新统计
        self.stats_label.configure(text=f"已收集: {len(self.feedback_data)} 条")
        
        print(f"已保存反馈 (共{len(self.feedback_data)}条)")
        
        # 生成新图片
        self.generate_image()
    
    def skip_image(self):
        """跳过当前图片"""
        self.generate_image()


def main():
    root = tk.Tk()
    app = GANFeedbackGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
