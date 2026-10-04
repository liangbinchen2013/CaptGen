import pandas as pd
import matplotlib.pyplot as plt

# ===================== 配置 =====================
csv_path = "output\\training_log.csv"   # 修改成你的csv文件路径
sample_step = 10         # 每隔10行采样，线条不拥挤
# ================================================

df = pd.read_csv(csv_path)
df = df[::sample_step].copy()  # 采样

# 创建画布，4张子图 (v7: +Diversity)
fig, axes = plt.subplots(4, 1, figsize=(14, 16), dpi=120)

# 子图1：各类Loss
ax1 = axes[0]
ax1.plot(df["step"], df["d_loss"], label="D Loss", linewidth=0.9)
ax1.plot(df["step"], df["g_loss"], label="G Loss", linewidth=0.9)
ax1.plot(df["step"], df["aux_loss"], label="Aux Loss", linewidth=0.9)
ax1.plot(df["step"], df["fm_loss"], label="FM Loss", linewidth=0.9)
ax1.set_title("Loss Curves")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

# 子图2：正则项 R1、R2、contrast
ax2 = axes[1]
ax2.plot(df["step"], df["r1_penalty"], label="R1 Penalty", color="orange")
if "r2_penalty" in df.columns:
    ax2.plot(df["step"], df["r2_penalty"], label="R2 Penalty", color="purple", alpha=0.7)
ax2.plot(df["step"], df["contrast_loss"], label="Contrast Loss", color="green")
ax2.set_title("Regularization Terms")
ax2.set_ylabel("Value")
ax2.legend()
ax2.grid(alpha=0.3)

# 子图3：梯度范数
ax3 = axes[2]
ax3.plot(df["step"], df["d_grad_norm"], label="D Grad Norm", color="red")
ax3.plot(df["step"], df["g_grad_norm"], label="G Grad Norm", color="blue")
ax3.set_title("Gradient Norm")
ax3.set_xlabel("Step")
ax3.set_ylabel("Norm")
ax3.legend()
ax3.grid(alpha=0.3)

# 子图4：多样性损失 (v7: 逐位分组, 现在会有非零值)
ax4 = axes[3]
ax4.plot(df["step"], df["div_loss"], label="Diversity Loss", color="brown")
ax4.set_title("Diversity Loss (Position-Level)")
ax4.set_xlabel("Step")
ax4.set_ylabel("Value")
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_overview.png")
plt.show()
