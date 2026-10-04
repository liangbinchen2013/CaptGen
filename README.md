# 验证码生成 AI

[中文文档](README.md) | [English](README_EN.md)

基于 **AC-GAN**（条件生成对抗网络）的 4 位字母数字验证码生成器。
目标不是单纯"生成好看图片"，而是让生成结果同时满足：

1. **人类可读** —— 字形清晰、边缘锐利、背景干扰合理；
2. **机器画像贴近真实数据** —— 第三方 OCR（ddddocr / ppllocr）在生成图上的
   整图准确率、字符错误率、混淆分布与真实训练数据接近；
3. **同标签多样** —— 相同文本可生成不同噪声/背景的验证码，不过拟合标签。

---

## 效果展示

真实训练数据（上）与模型生成（下）：

![真实数据样例](docs/real_samples.png)

![生成数据样例](docs/generated_samples.png)

> 数据均为 35×90 灰度图，4 字符，字符集 `0-9 + A-Z`（36 类）。

---

## 训练结果

### 关键指标（400 epochs / 468,000 steps）

| 指标 | 真实数据 | 生成数据（训练日志） | 生成数据（独立评测 n=2000） |
|---|---|---|---|
| CaptchaResNet 整图准确率 | ~99%（训练集模型，仅作可读性代理） | **94.4%** | **94.6%** |
| ddddocr 整图准确率 | **79.8%** | **80.6%** | 79.6% |
| ppllocr 整图准确率 | **67.9%** | 62.7% | 59.7% |
| gap 墨迹比（字符粘连度） | 0.84 / 0.96 / 0.69 | **0.79** | 0.84 |
| 逐字符错误率 MAE | — | — | **0.045** |
| 越界预测率 | 0.066 / 0.067 | 0.074 / 0.041 | PASS |

ddddocr 对生成图的识别率与真实图几乎一致（约 80%），说明生成图在机器视角下
已高度接近真实验证码；同时 `M6 逐字符错误率 MAE < 0.05`，说明**错在哪些字符上**
也与真实分布基本一致。

### 里程碑评测（M1-M8）

`evaluate_generated.py` 将生成图画像与 `confusion_profile.json`（真实数据全量画像）
逐项对比，发布版结果：

```
PASS M1 ddddocr acc比 fake/real: 0.9965   target[0.85, 1.05]
PASS M2 ddddocr 字符错误率比:     0.8531   target[0.8, 1.3]
PASS M1 ppllocr acc比 fake/real: 0.8789   target[0.85, 1.05]
PASS M2 ppllocr 字符错误率比:     1.2355   target[0.8, 1.3]
INFO M3 混淆分布JS散度:          0.3767   (参考线<0.1, 越低越接近真实画像)
PASS M4 <LEN>错误比 (匹配真实):   0.73 / 1.22   target[0.5, 2.0]
PASS M5 越界率:                   0.074 / 0.041
PASS M6 逐字符错误率MAE:          0.0446   target[<0.05]
PASS M7 gap墨迹比 (匹配真实):     0.8388   target 真实均值±0.15
PASS M8 CaptchaResNet整图acc:     0.9460   target[>=0.85]
```

> M3 为参考指标：生成图的混淆分布与真实画像尚未完全重合，
> 可通过延长 Phase C（混淆软目标）训练进一步收敛。

### 训练 Loss 曲线

![Loss 曲线](docs/loss_curves.png)

![OCR / gap / 判别器曲线](docs/ocr_curves.png)

- **G/D Loss**：G 预热期仅辅助损失；对抗权重引入后 G Loss 上升属正常现象，
  随后稳定在 0.6 \~ 0.7；D Loss 稳定在 1.2 \~ 1.6。
- **视觉正则**：Edge / Realism / Diversity 均随训练收敛；Contrast 损失接近 0，
  说明生成图对比度已落在真实数据分布带内。
- **D Grad Norm** 后期升高但训练稳定（有梯度裁剪）；R1/R2 保持 O(1)。
- **OCR 曲线**：Phase A（Ep1-100）仅 CaptchaResNet 强引导；Ep100 起接入
  第三方 OCR，ddddocr 于 Ep400 达到 80.6%，与真实 79.8% 基本重合。
- **gap 墨迹比**：全程稳定在真实水平（0.79 \~ 0.84），无硬切间隙。

---

## 主要攻克点：标签噪声下生成可读验证码

### 1. 标签噪声的实证来源

训练集 149,888 张（599,552 个字符）经全量统计分析，噪声主要来自三处：

| 噪声类型 | 实测现象 | 对训练的破坏 |
|---|---|---|
| 大小写随机标注 | 同一字母的大写/小写随机出现，甚至同一标签内混用 | 同一张图对应多个矛盾标签，模型被迫在 62 类间"掷硬币" |
| 字符分布非均匀 | 字符 `0` 从未出现；`1/5/O` 仅约 5,700 次（均匀期望约 16,600） | 均匀采样会生成训练集从未教过的字符，质量不可控 |
| 标签视觉歧义 | 真实标注中存在 OCR 都难以区分的对（O/0、I/L/T/1、C/E 等 26 对实证混淆） | 强行要求模型"分清"等于直接拟合标注噪声 |

此外字符粘连、背景干扰线与手写风格使像素到标签本身就是多对一映射，
**"把标签洗干净"既不可能、也会丢掉真实数据的分布特性**。

### 2. 攻克策略：把噪声变成监督信号的一部分

1. **标签统一降噪**：全部映射为大写 36 类，直接消除大小写矛盾监督信号。
2. **真实错误画像作为黄金标准**：对全量真实数据跑 ddddocr + ppllocr，
   产出 `confusion_profile.json`——逐字符真实错误率 `error_rate`、
   有向混淆分布 `confusion_dist`、26 对实证混淆对，以及字符分布 `label_counts`。
   后续所有监督都以"真实机器在真实数据上的表现"为参照，而不是假设标签 100% 干净。
3. **96 格字符级证据权重表**（`adaptive_ocr.py`）：
   三个 OCR 按字符位给出证据强弱，决定该位的损失权重。
   即使标签本身有歧义，只要多个 OCR 一致读出某个字，该位仍获得梯度；
   全部未命中时保留地板权重，保证可读性梯度不会断。
   这样模型学的是"证据充分的读法"，而不是"猜测噪声标签"。
4. **Phase C 混淆软目标**：训练后期将 one-hot 目标替换为
   `y_soft = (1-ε)·δ + ε·confusion_dist`——模型不需要表现出超过
   真实标注可分辨力的能力，在与真实数据同样的模糊边界处"错得一样"。
5. **防 OCR 欺骗（Goodhart 定律）**：直接优化 OCR 准确率会诱导模型生成
   骗过特定 OCR 的对抗样本。解法是把欺骗阈值设在真实准确率 **+5%**，
   配合"高置信独苗"惩罚（只有 CaptchaResNet 强命中、第三方全未命中时降权），
   保证优化方向是"人读得清"，而不是"机器被骗过"。
6. **冷启动 + 课程学习**：D 预训练 → G 预热 → 渐进对抗；
   Phase A 用 CaptchaResNet 强引导字形，Phase B 再接入第三方 OCR 防欺骗，
   避免一上来就在噪声标签上让多模型互相"投票"。

### 3. 攻克效果

- 真实标注本身就难以区分的字符上，生成图表现出与真实数据同级、同向的错误分布：
  `M6 逐字符错误率 MAE = 0.045`（目标 < 0.05）；
- ddddocr 对生成图 80.6% vs 对真实图 79.8%，机器视角下几乎不可区分；
- 同时保持可读性守卫 `CaptchaResNet ≥ 85%`（实际 94.6%），
  没有为了"形似"而牺牲清晰度。

> 核心结论：在标签噪声下不追求"100% 读对"，而是让模型学会**与真实数据一致的
> 可读性与模糊边界**——这比清洗标签后训练出的"过于干净"的模型更接近真实分布。

---

## 模型结构

### Generator（约 6M 参数）

```
z (256) ─┐
         ├─► cond_proj(512) ─► Linear ─► 4×6 特征 ─► Bottleneck
labels ──┘                                          │
                                    Self-Attention ◄┘
                                            │
        CondUpBlock: 4×6 → 8×12 → 16×24    │ (FiLM 条件注入 + 可学习噪声)
                                            ▼
                          Upsample 35×90 ─► Conv×2 ─► NoiseInjection
                                            ▼
                                    Conv 1×1 ─► tanh ─► (B,1,35,90)
```

- 小初始化投影（512→6144）避免大矩阵记忆训练集统计；
- FiLM 在每级分辨率注入文本条件；
- 逐通道**可学习噪声增益**在训练与推理时都注入，保证同标签多样性；
- 无 Skip Connection，避免编码器过拟合特征被直接复用。

### Discriminator（AC-GAN）

- 4 层谱归一化卷积 + InstanceNorm + Minibatch StdDev；
- **逐位辅助分类头**：池化特征按宽度拆成 4 列，每列独立识别一个字符位置，
  强制空间分离字符、缓解粘连；
- ReACGAN 超球面投影（辅助头输入 L2 归一化）稳定 AC-GAN 训练。

### EMA

对生成器权重做指数移动平均（decay=0.999），同时同步 BN 统计量，
推理/采样默认使用 EMA 权重。

---

## 训练策略

### 损失函数

| 损失 | 作用 |
|---|---|
| Hinge 对抗 | 判别真假，配合 SpectralNorm 约束 Lipschitz |
| R1 / R2 梯度惩罚 | 平衡真假数据上的判别器梯度，稳定训练 |
| Aux 逐位分类 | 生成器学会写出正确字符 |
| FM 特征匹配 | 生成图在判别器中间层贴近真实特征分布 |
| Edge（Sobel） | 约束锐利笔画，防模糊 |
| Contrast（双向） | 逐图 std 落入真实分布带，过高/过低都惩罚 |
| Realism | 匹配像素均值/标准差/梯度统计 |
| Diversity | 同标签双噪声 L1 距离下限，防模式坍塌 |
| OCR（课程学习） | ResNet 强引导 → 三 OCR 字符级证据加权，防欺骗 |

### OCR 课程学习

- **Phase A（Ep1-100）**：仅 CaptchaResNet 强引导字形，第三方 OCR 不加载；
- **Ramp（5 ep）**：A→B 线性混合；
- **Phase B**：ddddocr + ppllocr + CaptchaResNet 三方字符级证据权重表
  （96 格）逐位加权；任一 OCR 识别率显著高于真实基准则视为"欺骗"并下调权重；
- **Phase C（Ep300-400）**：以真实机器混淆画像（`confusion_profile.json`）
  作为软目标，让 OCR 的错法逐步接近真实数据。

### 自适应判别器增强（StyleGAN2-ADA）

对判别器输入施加平移/缩放/亮度/噪声/Cutout 等验证码安全增强
（不做水平翻转），增强概率按 D 对真实数据的正确率自适应调整。

---

## 快速开始

### 环境依赖

```bash
pip install -r requirements.txt
```

- CUDA GPU / CPU / 华为昇腾 NPU 均可运行（`device_utils.py` 自动检测）；
- 第三方 OCR（ddddocr、ppllocr）仅训练 Phase B 与评测时需要，
  缺失时训练可运行（自动降级为 CaptchaResNet 监督）。

### 数据准备

训练数据托管在 Hugging Face：
**https://huggingface.co/datasets/liangbinchen2013/CaptGen**

下载仓库中的 `data.zip` 并在项目根目录解压，即可得到 `训练数据/`
（149,888 张验证码，每 500 张一个 `batch_*` 目录）：

```bash
# 下载 (需要 pip install huggingface_hub)
huggingface-cli download liangbinchen2013/CaptGen data.zip --repo-type dataset --local-dir .

# 解压 (Windows 10+ 可用 tar, 或直接用资源管理器解压)
tar -xf data.zip
```

> 若只持有原始 Parquet 文件，可运行 `python extract_parquet.py`
> 生成同样结构的 `训练数据/`。

（可选）统计真实数据 OCR 错误画像，重新生成 `confusion_profile.json`：

```bash
python OCR_confusion_finding.py        # 全量跑 ddddocr / ppllocr, 生成 ocr_error_stats_*.txt
python build_confusion_profile.py      # 解析统计, 生成 confusion_profile.json
```

> 仓库已内置生成好的 `confusion_profile.json` 与两份 `ocr_error_stats_*.txt`。

### 训练

```bash
# 默认配置 (GPU)
python train.py

# 指定数据/输出目录与轮数
python train.py --data-dir ./训练数据 --output-dir ./output_v28 --epochs 400

# 断点续训 (自动恢复优化器与学习率调度器状态)
python train.py --resume output_v28/checkpoint_epoch_100.pt

# 关闭 OCR 课程学习 / 调整 Phase C / 开启反粘连实验
python train.py --no-curriculum --phase-c-epoch 300 --lambda-gap 0.05
```

训练过程会输出到 `OUTPUT_DIR`：

- `checkpoint_epoch_XXX.pt`：模型 + 优化器 + EMA + 调度器状态；
- `epoch_XXX_step_XXXXXX_{ema,train}.png`：固定噪声采样对比；
- `training_log.csv`：逐步 / 逐 epoch 的完整指标日志。

### 推理生成

```bash
python generate.py                                            # 自动选择最新 checkpoint
python generate.py --count 100 --output generated
python generate.py --text AB12 XY78                           # 指定文本
python generate.py --checkpoint output_v28/checkpoint_epoch_400.pt
python generate.py --label-dist uniform                       # 均匀标签 (默认为真实分布)
```

### 评测

```bash
# 综合质量评估 (像素统计 / OCR 可读性 / 多样性)
python evaluate.py --checkpoint output_v28/checkpoint_epoch_400.pt

# 里程碑评测: 生成图机器画像 vs 真实画像 (M1-M8)
python evaluate_generated.py --ckpt output_v28/checkpoint_epoch_400.pt --n 2000 --tag release

# 多样性专项检查
python verify_diversity.py --checkpoint output_v28/checkpoint_epoch_400.pt
```

### 单元测试（纯 CPU，不加载模型）

```bash
python test_adaptive_ocr.py
```

覆盖：96 格权重表结构与单调性、混淆对、字符串对齐、状态机、
课程学习/Phase C 端点、软目标 CE 数学性质、gap hinge 损失。

### 查看数据与生成结果

```bash
python visualize.py --samples                       # 训练集样本预览
python visualize.py --generated --input-dir generated
python analyze_data.py                              # 训练集详细统计报告
python analyze_training_log.py                      # 训练日志 Markdown 报告
python create_charts.py                             # 重新生成 docs/ 下的曲线图
python human_feedback_gui.py                        # 人工反馈收集 GUI (需图形界面)
```

---

## 项目结构

```
├── README.md                  # 中文说明 (本文件)
├── README_EN.md               # English documentation
├── config.py                  # 全局配置 (数据/模型/损失/课程学习/设备自适应)
├── models.py                  # Generator / Discriminator / EMA
├── train.py                   # 训练入口 (D 预训练 → G 预热 → 对抗 → 课程学习)
├── generate.py                # 推理生成
├── evaluate.py                # 综合质量评估
├── evaluate_generated.py      # 里程碑评测 (生成画像 vs 真实画像)
├── verify_diversity.py        # 生成多样性验证
├── adaptive_ocr.py            # 三 OCR 字符级证据权重表 + 自适应权重
├── ocr_net.py                 # CaptchaResNet (字形监督 / 可读性代理)
├── data_loader.py             # 数据集与 DataLoader
├── device_utils.py            # NPU / CUDA / CPU 自适应
├── utils.py                   # 采样保存 / 模型加载
├── create_charts.py           # 训练曲线绘图 (docs/)
├── analyze_data.py            # 数据集分析
├── analyze_training_log.py    # 训练日志分析
├── human_feedback_gui.py      # 人工反馈 GUI
├── OCR_confusion_finding.py   # 真实数据 OCR 错误统计
├── build_confusion_profile.py # 生成机器混淆画像 JSON
├── extract_parquet.py         # Parquet → 训练数据 JPG
├── test_adaptive_ocr.py       # 核心逻辑单元测试
├── confusion_profile.json     # 真实数据机器混淆画像 (资产)
├── ocr_error_stats_*.txt      # 双 OCR 全量错误统计 (中间产物)
├── OCR/best_captcha_resnet.pth# CaptchaResNet 权重
├── docs/                      # README 配图 (曲线 / 样例)
├── data.zip                   # 训练数据压缩包 (来自 Hugging Face)
├── 训练数据/batch_*/          # 训练集 (149,888 张, data.zip 解压结果)
└── output_v28/                # 训练输出 (checkpoint / 样例 / 日志)
```

---

## 说明与注意事项

- **标签大小写**：原始标注大小写随机且不可靠，训练统一映射为大写，
  以消除矛盾监督信号；
- **字符分布不等**：训练集中字符 `0` 缺失，`1/5/O` 等较稀有；
  `generate.py` 默认按真实标签分布采样（`--label-dist uniform` 可切换）；
- **反粘连 gap 损失默认关闭**：历史实验证明无下界的 gap 最小化会把字符间
  压出硬间隙并截断笔画；`--lambda-gap` 可开启带上限 hinge 的实验模式；
- **D Aux 准确率会接近 100%**：这是判别器自证指标，不能作为质量依据，
  请以 CaptchaResNet / 第三方 OCR 指标为准；
- **磁盘占用**：`output_v28/` 含每 10 epoch 的 checkpoint（每个约 115MB），
  发布/部署只需保留最终 `checkpoint_epoch_400.pt` 与 `training_log.csv`；
- **性能参考**：RTX 3070、batch=128 训练 400 epochs 约 4.5 天，
  约 1.2 step/s；NPU（昇腾 910B3）会自动切换到 batch=1024。

## 参考

- SNGAN / Hinge Loss: Miyato et al., 2018
- StyleGAN2-ADA: Karras et al., NeurIPS 2020
- R1 + R2 正则: Mescheder et al., 2018; R3GAN, Huang et al., NeurIPS 2024
- ReACGAN: Kang et al., NeurIPS 2021
- Minibatch StdDev: Progressive GANs, Karras et al., 2018

