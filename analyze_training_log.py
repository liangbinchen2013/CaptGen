"""
训练日志详细分析 — 输出 Markdown 报告

读取 output_v28/training_log.csv, 统计各阶段损失/判别器平衡/梯度稳定性,
并给出训练质量评估与建议。
"""
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_training_log():
    csv_path = Path(__file__).resolve().parent / "output_v28" / "training_log.csv"
    df = pd.read_csv(csv_path)
    
    lines = []
    def p(s=""):
        lines.append(s)
    
    p("=" * 80)
    p("  验证码生成 AI 训练日志详细分析报告")
    p("=" * 80)
    
    # 1. 总体统计
    p("\n" + "─" * 80)
    p("  1. 训练总体统计")
    p("─" * 80)
    
    total_steps = len(df)
    epochs = df["epoch"].max()
    start_time = pd.to_datetime(df["timestamp"].iloc[0])
    end_time = pd.to_datetime(df["timestamp"].iloc[-1])
    duration = end_time - start_time
    
    p(f"  总训练步数:      {total_steps}")
    p(f"  总 Epoch 数:     {epochs}")
    p(f"  训练开始时间:    {start_time}")
    p(f"  训练结束时间:    {end_time}")
    p(f"  总训练时长:      {duration}")
    p(f"  平均每步耗时:    {duration.total_seconds() / total_steps:.2f} 秒")
    
    phases = df["phase"].unique()
    p(f"\n  训练阶段:")
    for phase in phases:
        phase_df = df[df["phase"] == phase]
        phase_steps = len(phase_df)
        phase_epochs = phase_df["epoch"].nunique()
        p(f"    {phase:20s}: {phase_steps:>6} 步, {phase_epochs} 个 epoch")
    
    # 2. Loss 曲线分析
    p("\n" + "─" * 80)
    p("  2. Loss 曲线分析")
    p("─" * 80)
    
    train_df = df[df["step"] > 0].copy()
    
    loss_columns = ["d_loss", "g_loss", "aux_loss", "fm_loss", "contrast_loss", 
                    "div_loss", "edge_loss", "realism_loss", "r1_penalty", "r2_penalty"]
    
    p(f"\n  {'Loss 名称':<20} {'初始值':>10} {'最终值':>10} {'最小值':>10} {'最大值':>10} {'变化':>10}")
    p(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    
    for col in loss_columns:
        if col in train_df.columns:
            initial = train_df[col].iloc[:10].mean()
            final = train_df[col].iloc[-100:].mean()
            min_val = train_df[col].min()
            max_val = train_df[col].max()
            change = (final - initial) / initial * 100 if initial != 0 else 0
            p(f"  {col:<20} {initial:>10.4f} {final:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {change:>+9.1f}%")
    
    # 3. 判别器分析
    p("\n" + "─" * 80)
    p("  3. 判别器 (D) 分析")
    p("─" * 80)
    
    d_real_initial = train_df["d_real_mean"].iloc[:10].mean()
    d_fake_initial = train_df["d_fake_mean"].iloc[:10].mean()
    d_real_final = train_df["d_real_mean"].iloc[-100:].mean()
    d_fake_final = train_df["d_fake_mean"].iloc[-100:].mean()
    
    d_gap_initial = d_real_initial - d_fake_initial
    d_gap_final = d_real_final - d_fake_final
    
    p(f"\n  D 输出分析:")
    p(f"    初始阶段 (前10步):")
    p(f"      d_real_mean:    {d_real_initial:+.4f}")
    p(f"      d_fake_mean:    {d_fake_initial:+.4f}")
    p(f"      d_gap:          {d_gap_initial:+.4f}")
    p(f"    最终阶段 (后100步):")
    p(f"      d_real_mean:    {d_real_final:+.4f}")
    p(f"      d_fake_mean:    {d_fake_final:+.4f}")
    p(f"      d_gap:          {d_gap_final:+.4f}")
    
    p(f"\n  D 判别能力评估:")
    if d_gap_final > 1.0:
        p(f"    [优秀] D gap = {d_gap_final:.2f} > 1.0, D 能有效区分真假")
    elif d_gap_final > 0.5:
        p(f"    [良好] D gap = {d_gap_final:.2f}, D 有一定判别力")
    elif d_gap_final > 0:
        p(f"    [警告] D gap = {d_gap_final:.2f}, D 判别力偏弱")
    else:
        p(f"    [危险] D gap = {d_gap_final:.2f} < 0, D 已崩溃!")
    
    p(f"\n  D 输出分布:")
    p(f"    d_real_mean 范围: [{train_df['d_real_mean'].min():+.4f}, {train_df['d_real_mean'].max():+.4f}]")
    p(f"    d_fake_mean 范围: [{train_df['d_fake_mean'].min():+.4f}, {train_df['d_fake_mean'].max():+.4f}]")
    
    # 4. 梯度分析
    p("\n" + "─" * 80)
    p("  4. 梯度分析")
    p("─" * 80)
    
    d_grad_initial = train_df["d_grad_norm"].iloc[:100].mean()
    d_grad_final = train_df["d_grad_norm"].iloc[-100:].mean()
    g_grad_initial = train_df["g_grad_norm"].iloc[:100].mean()
    g_grad_final = train_df["g_grad_norm"].iloc[-100:].mean()
    
    p(f"\n  梯度范数:")
    p(f"    D 梯度范数:  初始={d_grad_initial:.2f}, 最终={d_grad_final:.2f}")
    p(f"    G 梯度范数:  初始={g_grad_initial:.2f}, 最终={g_grad_final:.2f}")
    
    d_grad_std = train_df["d_grad_norm"].std()
    g_grad_std = train_df["g_grad_norm"].std()
    
    p(f"\n  梯度稳定性:")
    p(f"    D 梯度标准差: {d_grad_std:.2f} {'(稳定)' if d_grad_std < 2 else '(波动较大)'}")
    p(f"    G 梯度标准差: {g_grad_std:.2f} {'(稳定)' if g_grad_std < 2 else '(波动较大)'}")
    
    # 5. 学习率分析
    p("\n" + "─" * 80)
    p("  5. 学习率分析")
    p("─" * 80)
    
    lr_g_values = train_df["lr_g"].unique()
    p(f"\n  G 学习率变化: {len(lr_g_values)} 个不同值")
    for lr in sorted(lr_g_values, reverse=True):
        steps_at_lr = len(train_df[train_df["lr_g"] == lr])
        p(f"    {lr:.2e}: {steps_at_lr} 步")
    
    # 6. 分类准确率分析
    p("\n" + "─" * 80)
    p("  6. 分类准确率分析")
    p("─" * 80)
    
    aux_acc_real_initial = train_df["aux_acc_real"].iloc[:100].mean()
    aux_acc_real_final = train_df["aux_acc_real"].iloc[-100:].mean()
    aux_acc_fake_initial = train_df["aux_acc_fake"].iloc[:100].mean()
    aux_acc_fake_final = train_df["aux_acc_fake"].iloc[-100:].mean()
    
    p(f"\n  Aux 分类准确率:")
    p(f"    真实数据:  初始={aux_acc_real_initial*100:.2f}%, 最终={aux_acc_real_final*100:.2f}%")
    p(f"    生成数据:  初始={aux_acc_fake_initial*100:.2f}%, 最终={aux_acc_fake_final*100:.2f}%")
    
    p(f"\n  G 学习效果:")
    if aux_acc_fake_final > 0.9:
        p(f"    [优秀] 生成图 Aux 准确率 = {aux_acc_fake_final*100:.1f}% > 90%")
    elif aux_acc_fake_final > 0.7:
        p(f"    [良好] 生成图 Aux 准确率 = {aux_acc_fake_final*100:.1f}%")
    else:
        p(f"    [警告] 生成图 Aux 准确率 = {aux_acc_fake_final*100:.1f}%")
    
    # 7. 各阶段详细分析
    p("\n" + "─" * 80)
    p("  7. 各阶段详细分析")
    p("─" * 80)
    
    for phase in ["WARM", "ADV_WARM", "JOINT"]:
        phase_df = train_df[train_df["phase"] == phase]
        if len(phase_df) == 0:
            continue
            
        p(f"\n  [{phase}] 阶段:")
        p(f"    步数范围:    {phase_df['step'].min()} - {phase_df['step'].max()}")
        p(f"    Epoch 范围:  {phase_df['epoch'].min()} - {phase_df['epoch'].max()}")
        p(f"    D Loss:      {phase_df['d_loss'].mean():.4f} +/- {phase_df['d_loss'].std():.4f}")
        p(f"    G Loss:      {phase_df['g_loss'].mean():.4f} +/- {phase_df['g_loss'].std():.4f}")
        p(f"    Aux Loss:    {phase_df['aux_loss'].mean():.4f} +/- {phase_df['aux_loss'].std():.4f}")
        p(f"    FM Loss:     {phase_df['fm_loss'].mean():.4f} +/- {phase_df['fm_loss'].std():.4f}")
        p(f"    Edge Loss:   {phase_df['edge_loss'].mean():.4f} +/- {phase_df['edge_loss'].std():.4f}")
        p(f"    Realism Loss:{phase_df['realism_loss'].mean():.4f} +/- {phase_df['realism_loss'].std():.4f}")
        p(f"    Div Loss:    {phase_df['div_loss'].mean():.4f} +/- {phase_df['div_loss'].std():.4f}")
        p(f"    D Gap:       {(phase_df['d_real_mean'] - phase_df['d_fake_mean']).mean():.4f}")
        p(f"    Aux Acc Real:{phase_df['aux_acc_real'].mean()*100:.2f}%")
        p(f"    Aux Acc Fake:{phase_df['aux_acc_fake'].mean()*100:.2f}%")
    
    # 8. 关键指标趋势
    p("\n" + "─" * 80)
    p("  8. 关键指标趋势 (每50 epoch)")
    p("─" * 80)
    
    p(f"\n  {'Epoch':>6} {'D Loss':>10} {'G Loss':>10} {'D Gap':>10} {'Aux Acc':>10} {'Grad Norm':>10}")
    p(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    
    for epoch in range(50, epochs + 1, 50):
        epoch_df = train_df[(train_df["epoch"] == epoch) & (train_df["step"] > 0)]
        if len(epoch_df) == 0:
            continue
        
        d_loss = epoch_df["d_loss"].mean()
        g_loss = epoch_df["g_loss"].mean()
        d_gap = (epoch_df["d_real_mean"] - epoch_df["d_fake_mean"]).mean()
        aux_acc = epoch_df["aux_acc_fake"].mean()
        grad_norm = epoch_df["g_grad_norm"].mean()
        
        p(f"  {epoch:>6} {d_loss:>10.4f} {g_loss:>10.4f} {d_gap:>+10.4f} {aux_acc*100:>9.2f}% {grad_norm:>10.2f}")
    
    # 9. 详细阶段数据
    p("\n" + "─" * 80)
    p("  9. 各 epoch 关键指标 (每10 epoch)")
    p("─" * 80)
    
    p(f"\n  {'Epoch':>6} {'Phase':<12} {'D Loss':>8} {'G Loss':>8} {'D Real':>8} {'D Fake':>8} {'D Gap':>8} {'Aux%':>7}")
    p(f"  {'─'*6} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    
    for epoch in range(1, epochs + 1, 10):
        epoch_df = train_df[(train_df["epoch"] == epoch) & (train_df["step"] > 0)]
        if len(epoch_df) == 0:
            continue
        
        d_loss = epoch_df["d_loss"].mean()
        g_loss = epoch_df["g_loss"].mean()
        d_real = epoch_df["d_real_mean"].mean()
        d_fake = epoch_df["d_fake_mean"].mean()
        d_gap = d_real - d_fake
        aux_acc = epoch_df["aux_acc_fake"].mean()
        phase = epoch_df["phase"].iloc[0]
        
        p(f"  {epoch:>6} {phase:<12} {d_loss:>8.4f} {g_loss:>8.4f} {d_real:>+8.4f} {d_fake:>+8.4f} {d_gap:>+8.4f} {aux_acc*100:>6.1f}%")
    
    # 10. 训练质量评估
    p("\n" + "═" * 80)
    p("  10. 训练质量综合评估")
    p("═" * 80)
    
    scores = {
        "D 稳定性": 0,
        "G 学习效果": 0,
        "梯度稳定性": 0,
        "收敛性": 0,
        "分类能力": 0,
    }
    
    # D 稳定性评分
    if d_gap_final > 1.0:
        scores["D 稳定性"] = 100
    elif d_gap_final > 0.5:
        scores["D 稳定性"] = 80
    elif d_gap_final > 0:
        scores["D 稳定性"] = 60
    else:
        scores["D 稳定性"] = 20
    
    # G 学习效果评分
    if aux_acc_fake_final > 0.9:
        scores["G 学习效果"] = 100
    elif aux_acc_fake_final > 0.8:
        scores["G 学习效果"] = 85
    elif aux_acc_fake_final > 0.7:
        scores["G 学习效果"] = 70
    else:
        scores["G 学习效果"] = 50
    
    # 梯度稳定性评分
    if d_grad_std < 1:
        scores["梯度稳定性"] = 100
    elif d_grad_std < 2:
        scores["梯度稳定性"] = 80
    elif d_grad_std < 3:
        scores["梯度稳定性"] = 60
    else:
        scores["梯度稳定性"] = 40
    
    # 收敛性评分
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
    
    # 分类能力评分
    scores["分类能力"] = int(aux_acc_fake_final * 100)
    
    p(f"\n  评分明细:")
    for name, score in scores.items():
        bar = "#" * (score // 5) + "-" * (20 - score // 5)
        p(f"    {name:<12}: [{bar}] {score}/100")
    
    total_score = sum(scores.values()) / len(scores)
    p(f"\n  综合评分: {total_score:.1f}/100")
    
    if total_score >= 85:
        p(f"  评价: [优秀] 模型训练质量很高")
    elif total_score >= 70:
        p(f"  评价: [良好] 模型训练质量较好")
    elif total_score >= 55:
        p(f"  评价: [一般] 模型训练质量一般")
    else:
        p(f"  评价: [较差] 模型训练质量较差")
    
    # 11. 训练建议
    p("\n" + "─" * 80)
    p("  11. 训练建议")
    p("─" * 80)
    
    suggestions = []
    
    if d_gap_final < 0.5:
        suggestions.append("D gap 偏小，建议增加 D 更新次数或调整 D 学习率")
    
    if aux_acc_fake_final < 0.8:
        suggestions.append("生成图识别率偏低，建议增加训练轮数或调整 Aux 权重")
    
    if d_grad_std > 2 or g_grad_std > 2:
        suggestions.append("梯度波动较大，建议降低学习率或增加梯度裁剪")
    
    if train_df["edge_loss"].iloc[-100:].mean() > 0.1:
        suggestions.append("Edge Loss 偏高，生成图可能边缘模糊")
    
    if train_df["contrast_loss"].iloc[-100:].mean() > 0.05:
        suggestions.append("Contrast Loss 偏高，生成图对比度不足")
    
    if suggestions:
        for i, s in enumerate(suggestions, 1):
            p(f"  {i}. {s}")
    else:
        p(f"  训练过程正常，无需特别调整")
    
    p("\n" + "═" * 80)
    p("  分析完成")
    p("═" * 80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    report = analyze_training_log()
    
    # 保存到文件
    with open("training_log_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("Report saved to training_log_report.md")