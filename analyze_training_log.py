"""
训练日志详细分析脚本
分析300 epoch的完整训练过程
"""
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_training_log():
    csv_path = Path("output/training_log.csv")
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("  验证码生成AI v13 训练日志详细分析报告")
    print("=" * 80)
    
    # ═══════════════════════════════════════════════════════════
    #  1. 总体统计
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  1. 训练总体统计")
    print("─" * 80)
    
    total_steps = len(df)
    epochs = df["epoch"].max()
    start_time = pd.to_datetime(df["timestamp"].iloc[0])
    end_time = pd.to_datetime(df["timestamp"].iloc[-1])
    duration = end_time - start_time
    
    print(f"  总训练步数:      {total_steps}")
    print(f"  总 Epoch 数:     {epochs}")
    print(f"  训练开始时间:    {start_time}")
    print(f"  训练结束时间:    {end_time}")
    print(f"  总训练时长:      {duration}")
    print(f"  平均每步耗时:    {duration.total_seconds() / total_steps:.2f} 秒")
    
    # 训练阶段
    phases = df["phase"].unique()
    print(f"\n  训练阶段:        {', '.join(phases)}")
    
    for phase in phases:
        phase_df = df[df["phase"] == phase]
        phase_steps = len(phase_df)
        phase_epochs = phase_df["epoch"].nunique()
        print(f"    {phase:15s}: {phase_steps:>6} 步, {phase_epochs} 个 epoch")
    
    # ═══════════════════════════════════════════════════════════
    #  2. Loss 曲线分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  2. Loss 曲线分析")
    print("─" * 80)
    
    # 排除 EPOCH 日志行（step=0的行）
    train_df = df[df["step"] > 0].copy()
    
    loss_columns = ["d_loss", "g_loss", "aux_loss", "fm_loss", "contrast_loss", 
                    "div_loss", "edge_loss", "realism_loss"]
    
    print(f"\n  {'Loss 名称':<20} {'初始值':>10} {'最终值':>10} {'最小值':>10} {'最大值':>10} {'变化':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    
    for col in loss_columns:
        if col in train_df.columns:
            initial = train_df[col].iloc[:10].mean()
            final = train_df[col].iloc[-100:].mean()
            min_val = train_df[col].min()
            max_val = train_df[col].max()
            change = (final - initial) / initial * 100 if initial != 0 else 0
            print(f"  {col:<20} {initial:>10.4f} {final:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {change:>+9.1f}%")
    
    # ═══════════════════════════════════════════════════════════
    #  3. 判别器分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  3. 判别器 (D) 分析")
    print("─" * 80)
    
    # d_real_mean 和 d_fake_mean
    d_real_initial = train_df["d_real_mean"].iloc[:10].mean()
    d_fake_initial = train_df["d_fake_mean"].iloc[:10].mean()
    d_real_final = train_df["d_real_mean"].iloc[-100:].mean()
    d_fake_final = train_df["d_fake_mean"].iloc[-100:].mean()
    
    d_gap_initial = d_real_initial - d_fake_initial
    d_gap_final = d_real_final - d_fake_final
    
    print(f"\n  D 输出分析:")
    print(f"    初始阶段 (前10步):")
    print(f"      d_real_mean:    {d_real_initial:+.4f}")
    print(f"      d_fake_mean:    {d_fake_initial:+.4f}")
    print(f"      d_gap:          {d_gap_initial:+.4f}")
    print(f"    最终阶段 (后100步):")
    print(f"      d_real_mean:    {d_real_final:+.4f}")
    print(f"      d_fake_mean:    {d_fake_final:+.4f}")
    print(f"      d_gap:          {d_gap_final:+.4f}")
    
    # D 判别能力评估
    print(f"\n  D 判别能力评估:")
    if d_gap_final > 1.0:
        print(f"    [优秀] D gap = {d_gap_final:.2f} > 1.0, D 能有效区分真假")
    elif d_gap_final > 0.5:
        print(f"    [良好] D gap = {d_gap_final:.2f}, D 有一定判别力")
    elif d_gap_final > 0:
        print(f"    [警告] D gap = {d_gap_final:.2f}, D 判别力偏弱")
    else:
        print(f"    [危险] D gap = {d_gap_final:.2f} < 0, D 已崩溃!")
    
    # D Real/Fake 分布
    print(f"\n  D 输出分布:")
    print(f"    d_real_mean 范围: [{train_df['d_real_mean'].min():+.4f}, {train_df['d_real_mean'].max():+.4f}]")
    print(f"    d_fake_mean 范围: [{train_df['d_fake_mean'].min():+.4f}, {train_df['d_fake_mean'].max():+.4f}]")
    
    # ═══════════════════════════════════════════════════════════
    #  4. 梯度分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  4. 梯度分析")
    print("─" * 80)
    
    d_grad_initial = train_df["d_grad_norm"].iloc[:100].mean()
    d_grad_final = train_df["d_grad_norm"].iloc[-100:].mean()
    g_grad_initial = train_df["g_grad_norm"].iloc[:100].mean()
    g_grad_final = train_df["g_grad_norm"].iloc[-100:].mean()
    
    print(f"\n  梯度范数:")
    print(f"    D 梯度范数:  初始={d_grad_initial:.2f}, 最终={d_grad_final:.2f}")
    print(f"    G 梯度范数:  初始={g_grad_initial:.2f}, 最终={g_grad_final:.2f}")
    
    # 梯度稳定性
    d_grad_std = train_df["d_grad_norm"].std()
    g_grad_std = train_df["g_grad_norm"].std()
    
    print(f"\n  梯度稳定性:")
    print(f"    D 梯度标准差: {d_grad_std:.2f} {'(稳定)' if d_grad_std < 2 else '(波动较大)'}")
    print(f"    G 梯度标准差: {g_grad_std:.2f} {'(稳定)' if g_grad_std < 2 else '(波动较大)'}")
    
    # ═══════════════════════════════════════════════════════════
    #  5. 学习率分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  5. 学习率分析")
    print("─" * 80)
    
    lr_g_values = train_df["lr_g"].unique()
    lr_d_values = train_df["lr_d"].unique()
    
    print(f"\n  G 学习率变化: {len(lr_g_values)} 个不同值")
    for lr in sorted(lr_g_values, reverse=True):
        steps_at_lr = len(train_df[train_df["lr_g"] == lr])
        print(f"    {lr:.2e}: {steps_at_lr} 步")
    
    print(f"\n  D 学习率变化: {len(lr_d_values)} 个不同值")
    for lr in sorted(lr_d_values, reverse=True):
        steps_at_lr = len(train_df[train_df["lr_d"] == lr])
        print(f"    {lr:.2e}: {steps_at_lr} 步")
    
    # ═══════════════════════════════════════════════════════════
    #  6. 分类准确率分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  6. 分类准确率分析")
    print("─" * 80)
    
    # Aux 准确率
    aux_acc_real_initial = train_df["aux_acc_real"].iloc[:100].mean()
    aux_acc_real_final = train_df["aux_acc_real"].iloc[-100:].mean()
    aux_acc_fake_initial = train_df["aux_acc_fake"].iloc[:100].mean()
    aux_acc_fake_final = train_df["aux_acc_fake"].iloc[-100:].mean()
    
    print(f"\n  Aux 分类准确率:")
    print(f"    真实数据:  初始={aux_acc_real_initial*100:.2f}%, 最终={aux_acc_real_final*100:.2f}%")
    print(f"    生成数据:  初始={aux_acc_fake_initial*100:.2f}%, 最终={aux_acc_fake_final*100:.2f}%")
    
    # G 学习效果
    print(f"\n  G 学习效果:")
    if aux_acc_fake_final > 0.9:
        print(f"    [优秀] 生成图 Aux 准确率 = {aux_acc_fake_final*100:.1f}% > 90%")
    elif aux_acc_fake_final > 0.7:
        print(f"    [良好] 生成图 Aux 准确率 = {aux_acc_fake_final*100:.1f}%")
    elif aux_acc_fake_final > 0.5:
        print(f"    [一般] 生成图 Aux 准确率 = {aux_acc_fake_final*100:.1f}%")
    else:
        print(f"    [警告] 生成图 Aux 准确率 = {aux_acc_fake_final*100:.1f}% < 50%")
    
    # ═══════════════════════════════════════════════════════════
    #  7. 各阶段详细分析
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  7. 各阶段详细分析")
    print("─" * 80)
    
    for phase in ["WARM", "ADV_WARM", "JOINT"]:
        phase_df = train_df[train_df["phase"] == phase]
        if len(phase_df) == 0:
            continue
            
        print(f"\n  [{phase}] 阶段:")
        print(f"    步数范围:    {phase_df['step'].min()} - {phase_df['step'].max()}")
        print(f"    Epoch 范围:  {phase_df['epoch'].min()} - {phase_df['epoch'].max()}")
        print(f"    D Loss:      {phase_df['d_loss'].mean():.4f} ± {phase_df['d_loss'].std():.4f}")
        print(f"    G Loss:      {phase_df['g_loss'].mean():.4f} ± {phase_df['g_loss'].std():.4f}")
        print(f"    Aux Loss:    {phase_df['aux_loss'].mean():.4f} ± {phase_df['aux_loss'].std():.4f}")
        print(f"    FM Loss:     {phase_df['fm_loss'].mean():.4f} ± {phase_df['fm_loss'].std():.4f}")
        print(f"    D Gap:       {(phase_df['d_real_mean'] - phase_df['d_fake_mean']).mean():.4f}")
        print(f"    Aux Acc Real:{phase_df['aux_acc_real'].mean()*100:.2f}%")
        print(f"    Aux Acc Fake:{phase_df['aux_acc_fake'].mean()*100:.2f}%")
    
    # ═══════════════════════════════════════════════════════════
    #  8. 关键指标趋势
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  8. 关键指标趋势 (每50 epoch)")
    print("─" * 80)
    
    print(f"\n  {'Epoch':>6} {'D Loss':>10} {'G Loss':>10} {'D Gap':>10} {'Aux Acc':>10} {'Grad Norm':>10}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    
    for epoch in range(50, epochs + 1, 50):
        epoch_df = train_df[(train_df["epoch"] == epoch) & (train_df["step"] > 0)]
        if len(epoch_df) == 0:
            continue
        
        d_loss = epoch_df["d_loss"].mean()
        g_loss = epoch_df["g_loss"].mean()
        d_gap = (epoch_df["d_real_mean"] - epoch_df["d_fake_mean"]).mean()
        aux_acc = epoch_df["aux_acc_fake"].mean()
        grad_norm = epoch_df["g_grad_norm"].mean()
        
        print(f"  {epoch:>6} {d_loss:>10.4f} {g_loss:>10.4f} {d_gap:>+10.4f} {aux_acc*100:>9.2f}% {grad_norm:>10.2f}")
    
    # ═══════════════════════════════════════════════════════════
    #  9. 训练质量评估
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  9. 训练质量综合评估")
    print("═" * 80)
    
    # 评分标准
    scores = {
        "D 稳定性": 0,
        "G 学习效果": 0,
        "梯度稳定性": 0,
        "收敛性": 0,
        "分类能力": 0,
    }
    
    # D 稳定性评分 (0-100)
    if d_gap_final > 1.0:
        scores["D 稳定性"] = 100
    elif d_gap_final > 0.5:
        scores["D 稳定性"] = 80
    elif d_gap_final > 0:
        scores["D 稳定性"] = 60
    else:
        scores["D 稳定性"] = 20
    
    # G 学习效果评分 (0-100)
    if aux_acc_fake_final > 0.9:
        scores["G 学习效果"] = 100
    elif aux_acc_fake_final > 0.8:
        scores["G 学习效果"] = 85
    elif aux_acc_fake_final > 0.7:
        scores["G 学习效果"] = 70
    else:
        scores["G 学习效果"] = 50
    
    # 梯度稳定性评分 (0-100)
    if d_grad_std < 1:
        scores["梯度稳定性"] = 100
    elif d_grad_std < 2:
        scores["梯度稳定性"] = 80
    elif d_grad_std < 3:
        scores["梯度稳定性"] = 60
    else:
        scores["梯度稳定性"] = 40
    
    # 收敛性评分 (0-100)
    g_loss_change = (train_df["g_loss"].iloc[-100:].mean() - 
                     train_df["g_loss"].iloc[:100].mean())
    if g_loss_change < -0.5:
        scores["收敛性"] = 100
    elif g_loss_change < -0.2:
        scores["收敛性"] = 80
    elif g_loss_change < 0:
        scores["收敛性"] = 60
    else:
        scores["收敛性"] = 40
    
    # 分类能力评分 (0-100)
    scores["分类能力"] = int(aux_acc_fake_final * 100)
    
    print(f"\n  评分明细:")
    for name, score in scores.items():
        bar = "#" * (score // 5) + "-" * (20 - score // 5)
        print(f"    {name:<12}: [{bar}] {score}/100")
    
    total_score = sum(scores.values()) / len(scores)
    print(f"\n  综合评分: {total_score:.1f}/100")
    
    if total_score >= 85:
        print(f"  评价: [优秀] 模型训练质量很高")
    elif total_score >= 70:
        print(f"  评价: [良好] 模型训练质量较好")
    elif total_score >= 55:
        print(f"  评价: [一般] 模型训练质量一般")
    else:
        print(f"  评价: [较差] 模型训练质量较差")
    
    # ═══════════════════════════════════════════════════════════
    #  10. 训练建议
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  10. 训练建议")
    print("─" * 80)
    
    suggestions = []
    
    # D 稳定性建议
    if d_gap_final < 0.5:
        suggestions.append("D gap 偏小，建议增加 D 更新次数或调整 D 学习率")
    
    # G 学习效果建议
    if aux_acc_fake_final < 0.8:
        suggestions.append("生成图识别率偏低，建议增加训练轮数或调整 Aux 权重")
    
    # 梯度建议
    if d_grad_std > 2 or g_grad_std > 2:
        suggestions.append("梯度波动较大，建议降低学习率或增加梯度裁剪")
    
    # Loss 建议
    if train_df["edge_loss"].iloc[-100:].mean() > 0.1:
        suggestions.append("Edge Loss 偏高，生成图可能边缘模糊")
    
    if train_df["contrast_loss"].iloc[-100:].mean() > 0.1:
        suggestions.append("Contrast Loss 偏高，生成图对比度不足")
    
    if suggestions:
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s}")
    else:
        print(f"  训练过程正常，无需特别调整")
    
    print("\n" + "═" * 80)
    print("  分析完成")
    print("═" * 80)


if __name__ == "__main__":
    analyze_training_log()