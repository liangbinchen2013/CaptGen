# v16: 修复OCR欺骗问题

## 问题发现

v15训练中发现严重问题：
- `aux_acc_fake`从Epoch 50就达到**0.9996**（接近100%）
- 但生成的验证码**肉眼不可识读**
- Generator学会了"欺骗OCR"而非"生成可读字符"

## 问题根因

### 1. OCR Loss过强
```
v15: LAMBDA_OCR = 1.5  ← Generator过度优化这个目标
结果: Generator找到OCR漏洞，生成对抗样本
```

### 2. Contrast Loss失效
```
v15: CONTRAST_THRESHOLD = 0.15
实际: 生成图像std > 0.15，Contrast Loss始终=0
原因: 阈值过低，无法约束对比度
```

### 3. 单一OCR模型
```
只用一个CaptchaResNet作为监督
神经网络容易被对抗样本欺骗
Generator学到的是"欺骗这个特定OCR"
```

### 4. Goodhart定律
> "当一个指标变成目标时，它就不再是好的指标"

OCR准确率本是衡量可读性的好指标，但当Generator直接优化它时，就变成了"欺骗游戏"。

## 解决方案 (v16)

### 核心改动

| 参数 | v15 | v16 | 目的 |
|------|-----|-----|------|
| **LAMBDA_OCR** | 1.5 | **0.3** | 大幅降低，防止OCR欺骗 |
| **CONTRAST_THRESHOLD** | 0.15 | **0.25** | 提高到真实验证码水平 |
| **LAMBDA_CONTRAST** | 1.0 | **2.0** | 强化对比度约束 |

### 设计思路

1. **降低OCR权重**: 从1.5降到0.3，OCR仅作为辅助监督而非主导
2. **提高对比度要求**: 真实验证码std≈0.43-0.51，设阈值0.25
3. **强化视觉质量**: Edge Loss=2.0 + Contrast Loss=2.0 + Realism Loss=1.0

### 预期效果

| 指标 | v15现状 | v16预期 |
|------|---------|---------|
| aux_acc_fake | 0.9996 (欺骗) | 0.8-0.9 (真实) |
| Contrast Loss | 0.0000 (失效) | >0.01 (生效) |
| 人类可读性 | 差 | 显著改善 |

## 训练命令

```bash
python train.py --data-dir ./训练数据 --epochs 350 --use-cosine-lr --lr-min 5e-5
```

## 验证方法

训练时观察：
1. `aux_acc_fake`不应过快达到1.0
2. `contrast_loss`应>0（表示对比度约束生效）
3. 生成样例应肉眼可读

## 教训

**GAN训练中，过度优化单一指标会导致欺骗行为。**

正确做法：
- 多个损失函数平衡（Edge + Contrast + Realism + OCR）
- OCR权重不能过高（作为辅助而非主导）
- 视觉质量约束必须生效（Contrast/Edge Loss>0）
