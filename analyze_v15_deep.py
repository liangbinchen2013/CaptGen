import pandas as pd
import numpy as np
import os

# 使用相对路径
csv_path = os.path.join(os.getcwd(), 'output_v15', 'training_log.csv')
df = pd.read_csv(csv_path)

print('='*70)
print('  v15 深度分析: OCR欺骗问题')
print('='*70)

# 1. 各阶段指标对比
print('\n【1】各阶段关键指标:')
print('-'*70)
header = f"{'Epoch':<10} {'G Loss':<12} {'D Loss':<12} {'acc_fake':<12} {'Edge':<12} {'Contrast':<12}"
print(header)
print('-'*70)

for epoch in [1, 10, 30, 50, 100, 150, 200, 250]:
    edf = df[df['epoch'] == epoch]
    if len(edf) > 0:
        g = edf['g_loss'].mean()
        d = edf['d_loss'].mean()
        a = edf['aux_acc_fake'].mean()
        e = edf['edge_loss'].mean()
        c = edf['contrast_loss'].mean()
        print(f'{epoch:<10} {g:<12.3f} {d:<12.3f} {a:<12.4f} {e:<12.4f} {c:<12.6f}')

# 2. OCR欺骗时间点
print('\n【2】OCR欺骗时间点分析:')
print('-'*70)
acc_fake = df['aux_acc_fake']
first_100 = df[acc_fake >= 0.999].iloc[0] if len(df[acc_fake >= 0.999]) > 0 else None
if first_100 is not None:
    print(f'  首次达到0.999: Epoch {int(first_100["epoch"])}, Step {int(first_100["step"])}')

# 3. Contrast Loss分析
print('\n【3】Contrast Loss 分析:')
print('-'*70)
contrast_nonzero = df[df['contrast_loss'] > 0.001]
print(f'  Contrast Loss > 0.001 的步数: {len(contrast_nonzero)} / {len(df)} ({len(contrast_nonzero)/len(df)*100:.1f}%)')
if len(contrast_nonzero) > 0:
    print(f'  首次出现: Epoch {int(contrast_nonzero.iloc[0]["epoch"])}, Step {int(contrast_nonzero.iloc[0]["step"])}')
    print(f'  平均值: {contrast_nonzero["contrast_loss"].mean():.6f}')

# 4. Loss权重分析
print('\n【4】v15损失权重配置:')
print('-'*70)
print(f'  LAMBDA_OCR = 1.5      (OCR监督)')
print(f'  LAMBDA_CONTRAST = 1.0 (对比度)')
print(f'  LAMBDA_EDGE = 2.0     (边缘)')
print(f'  LAMBDA_REALISM = 1.0  (分布匹配)')
print(f'  CONTRAST_THRESHOLD = 0.15')

# 5. 问题诊断
print('\n【5】问题诊断:')
print('-'*70)
print('  [问题1] OCR Loss权重过高 (1.5)')
print('          -> Generator过度优化"欺骗OCR"而非"人类可读"')
print()
print('  [问题2] Contrast Loss完全失效')
print('          -> 生成图像std > 0.15, Loss始终=0')
print('          -> 对比度约束形同虚设')
print()
print('  [问题3] 单一OCR模型监督')
print('          -> CaptchaResNet容易被对抗样本欺骗')
print('          -> Generator学到的是"欺骗这个特定OCR"')
print()
print('  [问题4] Goodhart定律')
print('          -> "当指标变成目标时，就不再是好指标"')
print('          -> OCR准确率本是好指标，但直接优化导致欺骗')

# 6. 建议
print('\n【6】修复建议:')
print('-'*70)
print('  1. 降低OCR权重: 1.5 -> 0.3 (防止过度优化)')
print('  2. 提高Contrast阈值: 0.15 -> 0.25 (真实验证码std约0.43-0.51)')
print('  3. 强化Contrast权重: 1.0 -> 2.0 (强制高对比度)')
print('  4. 保持Edge/Realism权重 (视觉质量基础)')
