import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\ALIENWARE\\Desktop\\验证码生成AI\\output_v15\\training_log.csv')

print('='*60)
print('  v15 OCR欺骗问题分析')
print('='*60)

col = 'aux_acc_fake'
print(f'\n1. {col} 统计:')
print(f'   平均: {df[col].mean():.4f}')
print(f'   达到1.0的比例: {(df[col] >= 0.999).mean()*100:.1f}%')

print(f'\n2. 损失函数趋势:')
for epoch in [1, 50, 100, 150, 200]:
    edf = df[df['epoch'] == epoch]
    if len(edf) > 0:
        g = edf['g_loss'].mean()
        d = edf['d_loss'].mean()
        a = edf[col].mean()
        print(f'   Epoch {epoch}: G={g:.3f}, D={d:.3f}, acc_fake={a:.4f}')

print(f'\n3. Edge Loss趋势:')
for epoch in [1, 50, 100, 150, 200]:
    edf = df[df['epoch'] == epoch]
    if len(edf) > 0:
        e = edf['edge_loss'].mean()
        c = edf['contrast_loss'].mean()
        print(f'   Epoch {epoch}: Edge={e:.4f}, Contrast={c:.4f}')

print(f'\n4. d_fake_mean趋势 (越负=D越强):')
for epoch in [1, 50, 100, 150, 200]:
    edf = df[df['epoch'] == epoch]
    if len(edf) > 0:
        d = edf['d_fake_mean'].mean()
        print(f'   Epoch {epoch}: d_fake_mean={d:.4f}')
