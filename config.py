"""
验证码生成 AI —— 全局配置中心

数据集
  149,888 张 35×90 灰度验证码, 每张 4 个字符, 字符集 `0-9 + A-Z` (36 类)。
  原始标注大小写随机且不可靠, 统一映射为大写以消除矛盾监督信号。

模型与训练
  AC-GAN: 条件生成器 G + 判别器 D (对抗 + 逐位辅助分类), Hinge 对抗损失,
  R1/R2 梯度惩罚, StyleGAN2-ADA 自适应判别器增强, EMA 权重平滑。

关键机制
  1. OCR 课程学习
     Phase A: 仅用 CaptchaResNet 强引导字形可读性 (冷启动);
     Phase B: 引入 ddddocr/ppllocr 第三方 OCR 做防欺骗检测与字符级证据加权。
  2. 机器混淆画像 (confusion_profile.json)
     以第三方 OCR 在真实数据上的错误分布为"黄金标准", 后期 (Phase C)
     将监督目标软化为真实机器混淆分布, 让生成结果"机器错得像真实数据"。
  3. 视觉质量约束
     边缘、对比度、分布真实性、多样性、判别器特征匹配共同作用,
     防止只优化单一指标导致的欺骗或模式坍塌。

所有路径默认相对项目根目录, 可通过 CLI 参数 `--data-dir` / `--output-dir` 覆盖。
"""
from pathlib import Path

# ─── 数据 ────────────────────────────────────────────────
DATA_DIR = Path("训练数据")
IMG_HEIGHT = 35
IMG_WIDTH = 90

# 字符集: 仅保留 0-9 + A-Z = 36 类; 所有小写标签自动转为大写
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARS)  # 36
CAPTCHA_LENGTH = 4        # 验证码字符数

# 标签统一映射: a→A, b→B, ..., z→Z
# 原始标注大小写随机, 舍弃大小写信息是最安全的策略
LABEL_CASE_MAP = {}
for _c in "abcdefghijklmnopqrstuvwxyz":
    LABEL_CASE_MAP[_c] = _c.upper()

# ─── 训练 ────────────────────────────────────────────────
BATCH_SIZE = 128            # GPU 安全默认, NPU 自动覆盖为 1024
EPOCHS = 400                # 充分利用约 15 万张训练数据
LEARNING_RATE_G = 2e-4      # GPU 安全默认, NPU 自动覆盖为 5e-4
LEARNING_RATE_D = 2e-4
BETA1 = 0.0                 # SNGAN 标准配置, 配合 SpectralNorm + Hinge Loss
BETA2 = 0.9

# ─── 模型 ────────────────────────────────────────────────
LATENT_DIM = 256            # 噪声维度, 决定同标签采样的多样性空间
EMBED_DIM = 64              # 字符嵌入维度
DIS_FEATURES = 50           # 判别器基础通道数
NOISE_STD = 0.05            # 生成器噪声注入增益的初始值 (逐通道可学习)

# ─── 损失权重 ────────────────────────────────────────────
LAMBDA_AUX_G = 1.5          # G 的辅助分类损失权重
LAMBDA_AUX_D = 1.0          # D 的辅助分类损失权重
LAMBDA_R1 = 0.6             # R1 梯度惩罚权重 (真实数据)
LAMBDA_R2 = 0.6             # R2 梯度惩罚权重 (生成数据)
LAMBDA_DIVERSITY = 1.0      # 同标签双噪声对比多样性损失权重
ADV_LAMBDA_MAX = 0.55       # 对抗损失权重上限 (渐进引入)

# ─── Aux 分类器稳定性 ───────────────────────────────────
NORMALIZE_AUX = True        # ReACGAN 超球面投影: L2 归一化 Aux 头输入,
                            # 防止分类器梯度爆炸, 稳定 AC-GAN 训练

# ─── 视觉质量损失权重 ────────────────────────────────────
LAMBDA_FM = 1.0             # 判别器中间层特征匹配权重
LAMBDA_CONTRAST = 3.0       # 双向对比度匹配权重
CONTRAST_MARGIN = 0.05      # 真实分布带外扩 margin (带外才惩罚)
CONTRAST_LO_Q = 0.10        # 真实逐图 std 分布下分位
CONTRAST_HI_Q = 0.90        # 真实逐图 std 分布上分位
LAMBDA_EDGE = 2.0           # Sobel 边缘匹配权重
EDGE_LOSS_MIN = 0.10        # Edge Loss 低于此值时权重翻倍, 防止监督衰减
LAMBDA_REALISM = 1.0        # 像素均值/标准差/梯度统计匹配权重

# ─── OCR 字符级证据权重表 (状态机与查表见 adaptive_ocr.py) ─
# 三个 OCR (CaptchaResNet / ddddocr / ppllocr) 按字符位给出 6×4×4=96 格
# 组合权重; 全部未命中时保留地板权重, 避免"无梯度 → 永远读不出"的死锁。
LAMBDA_OCR = 0.15           # OCR 损失外层权重上限 (自适应调整)
OCR_CE_WEIGHT = 0.4         # OCR 损失中 CE 占比 (特征匹配占 0.6)
OCR_WARMUP_EPOCHS = 5       # G 预热结束后线性引入 OCR 损失的轮数

W_FLOOR = 0.08              # 全未命中/丢失位地板: 破死锁且梯度极弱
C_CONF_HI = 0.90            # CaptchaResNet 强命中置信度阈值
C_CONF_LO = 0.50            # CaptchaResNet 命中/弱命中分界
C_RANK2_PROB = 0.25         # 近失: 标签为第 2 名且概率不低于此值
M_LEN = 0.85                # 串长 ≠ 4 的结构折损 (D/P 各计一次)
M_LEN_MIN = 0.65            # M_LEN 叠加下限
M_CHEAT = 0.50              # "高置信独苗"惩罚: C 强命中且 D/P 均未命中
M_CONS = 1.10               # D==P 整串一致且 != 标签: 可读性共识奖励
M_CONS_CAP = 0.95           # 共识奖励后的终值上限
OCR_VALID_THR = 0.15        # 有效字符权重判定阈值 (统计有效占比用)

# ─── 机器混淆画像 (由 build_confusion_profile.py 生成) ────
# 真实数据上 ddddocr+ppllocr 的全量错误画像:
#   error_rate[c]        逐字符真实错误率
#   confusion_dist[c][p] 有向混淆分布 (软目标用, 36 类内归一化)
#   confusable_pairs     实证混淆对 (双 OCR 双向证据充分)
import json as _json
_confusion_profile = None
try:
    with open(Path(__file__).parent / "confusion_profile.json", encoding="utf-8") as _f:
        _confusion_profile = _json.load(_f)
except Exception:
    _confusion_profile = None

if _confusion_profile and _confusion_profile.get("confusable_pairs"):
    CONFUSABLE_PAIRS = tuple(tuple(p) for p in _confusion_profile["confusable_pairs"])
else:
    # 画像文件缺失时的保守回退 (仅最典型的视觉混淆对)
    CONFUSABLE_PAIRS = (("0", "O"), ("1", "I"), ("2", "Z"), ("5", "S"), ("8", "B"))

# Phase C 软目标原料 (缺失时 Phase C 自动退化为硬 CE)
CONFUSION_DIST = (_confusion_profile or {}).get("confusion_dist", {})
OCR_ERROR_RATE = (_confusion_profile or {}).get("error_rate", {})

# ─── Phase C: 混淆软目标 CE (Ep300-400) ──────────────────
# y_soft[c] = (1-ε_c)·δ_c + ε_c·confusion_dist[c], ε_c = min(error_rate[c], EPS_CAP)
# 目的: G 画的字符让 OCR "部分读错, 且错法与真实数据一致" (机器干扰)。
PHASE_C_EPOCHS = 300        # Phase C 起始轮 (--phase-c-epoch 可覆盖, 0=禁用)
PHASE_C_RAMP = 5            # ε 线性爬升轮数
EPS_CAP = 0.35              # ε 上限: 防止软目标过散损害可读性

# ─── 反粘连 gap 墨迹损失 (可选, 默认关闭) ─────────────────
# 字符列界带 (CaptchaResNet 四等分池化列界 x≈22.5/45/67.5) 内压笔画对比度。
# 真实数据列界带墨迹/峰值约为 0.84/0.96/0.69 (字符本就轻微粘连)。
# 注意: 无下界地最小化会把 gap 比压向 0, 切出硬间隙并截断笔画,
#       因此仅对超过上限的部分施加 hinge 惩罚, 且默认权重为 0。
LAMBDA_GAP = 0.0            # 0=关闭; >0 时启用上限 hinge
GAP_RATIO_MAX = 0.5         # 软 gap 比超过此值才惩罚
GAP_STRIPS = ((21, 25), (43, 47), (66, 70))   # 3 条列界带 [x0,x1)
GAP_INK_TAU = 0.15          # 墨迹判定: |灰度-背景| > tau 计为笔画

# ─── OCR 课程学习 ────────────────────────────────────────
# Phase A (Ep 1-CURRICULUM_EPOCHS): 仅 CaptchaResNet 强引导字形可读性;
#   第三方 OCR 不加载 (节省 CPU), 不做欺骗检测, 允许阶段性辅助分类偏高。
# Ramp   (随后 CURRICULUM_RAMP 轮): A→B 线性混合。
# Phase B (之后): 三 OCR 字符级证据权重表, 防欺骗。
USE_CURRICULUM = True
CURRICULUM_EPOCHS = 100     # Phase A 轮数 (--curriculum-epochs 可覆盖)
CURRICULUM_RAMP = 5         # 切换期轮数 (--curriculum-ramp 可覆盖)


def curriculum_phase(epoch: int, curriculum_epochs: int = None,
                     curriculum_ramp: int = None, enabled: bool = None):
    """
    返回 (phase, t):
        phase: 'A'=ResNet 强引导, 'AB'=切换期, 'B'=三 OCR 防欺骗
        t:     0.0=纯 A, 1.0=纯 B, 中间值=A/B 线性混合比例

    阶段只由 epoch 决定 (纯函数), 断点续训自动正确, checkpoint 无需额外字段。
    """
    if curriculum_epochs is None:
        curriculum_epochs = CURRICULUM_EPOCHS
    if curriculum_ramp is None:
        curriculum_ramp = CURRICULUM_RAMP
    if enabled is None:
        enabled = USE_CURRICULUM

    if not enabled:
        return "B", 1.0
    if epoch <= curriculum_epochs:
        return "A", 0.0
    if curriculum_ramp <= 0:
        return "B", 1.0
    t = (epoch - curriculum_epochs) / float(curriculum_ramp)
    if t >= 1.0:
        return "B", 1.0
    return "AB", t


def phase_c_t(epoch: int, phase_c_epochs: int = None,
              phase_c_ramp: int = None):
    """
    返回 (in_phase_c, t): Phase C 混淆软目标的 ε 爬升系数 (纯 epoch 函数)。
      t=0 → 硬 CE; t=1 → 全幅 ε_c; Ep300-305 线性爬升。
    """
    if phase_c_epochs is None:
        phase_c_epochs = PHASE_C_EPOCHS
    if phase_c_ramp is None:
        phase_c_ramp = PHASE_C_RAMP
    if phase_c_epochs <= 0:          # 0 = 禁用
        return False, 0.0
    if epoch < phase_c_epochs:
        return False, 0.0
    if phase_c_ramp <= 0:
        return True, 1.0
    t = (epoch - phase_c_epochs) / float(phase_c_ramp)
    return True, min(t, 1.0)


# ─── 训练策略 ────────────────────────────────────────────
D_UPDATES = 1                # D 基础更新次数 (按判别间隙自适应上调)
D_UPDATES_MAX = 3            # D 更新次数上限
D_ADAPTIVE_THRESHOLD = 0.3   # 判别间隙低于此值时增加 D 更新次数
D_PRETRAIN_EPOCHS = 3        # G 预热前先训练 D, 建立基础判别力
G_WARMUP_EPOCHS = 5          # G 纯辅助损失预热轮数
ADV_WARMUP_EPOCHS = 20       # 对抗权重线性爬升轮数
CLIP_NORM = 10.0             # 梯度裁剪上限

# ─── EMA ─────────────────────────────────────────────────
EMA_DECAY = 0.999            # 指数移动平均衰减系数 (窗口约 1000 步)

# ─── 学习率调度 ──────────────────────────────────────────
LR_MILESTONES = [150, 250]   # MultiStepLR 备选方案的衰减节点
LR_DECAY_FACTOR = 0.5
USE_COSINE_LR = True         # 默认余弦退火, 保持后期学习率活力
LR_MIN = 5e-5                # 余弦退火最小学习率

# ─── ADA (自适应判别器增强, StyleGAN2-ADA) ────────────────
USE_ADA = True               # 对 D 输入施加随机增强, 防止判别器过拟合
ADA_TARGET = 0.6             # D 对真实数据的正确率目标 (越高增强越强)
ADA_INTERVAL = 4             # 每 4 个 step 调整一次增强概率 p
ADA_P_INC = 2e-4             # 调整增量
ADA_P_DEC = 5e-5             # 调整减量 (降低更慢, 保守策略)

# ─── 标签平滑 ────────────────────────────────────────────
LABEL_SMOOTHING_EPS = 0.03

# ─── 保存与日志 ──────────────────────────────────────────
OUTPUT_DIR = Path("output_v28")   # 训练输出目录 (checkpoint / 样例 / 日志)
SAMPLE_INTERVAL = 1               # 样例生成间隔 (epoch)
MODEL_SAVE_INTERVAL = 10          # 模型保存间隔 (epoch)
LOG_INTERVAL = 20                 # 控制台日志间隔 (step)

# ─── 硬件 ────────────────────────────────────────────────
NUM_WORKERS = 4

# ═══════════════════════════════════════════════════════════
#  设备自适应 (必须在其他模块导入前执行)
# ═══════════════════════════════════════════════════════════
try:
    from device_utils import device_type
except ImportError:
    device_type = "cpu"

if device_type == "npu":
    # 华为昇腾 910B3, 90GB HBM
    BATCH_SIZE = 1024
    DIS_FEATURES = 96
    LEARNING_RATE_G = 5e-4
    LEARNING_RATE_D = 5e-4
    EPOCHS = 300
    NUM_WORKERS = 0
    LOG_INTERVAL = 20
    LR_MILESTONES = [150, 250]
    D_UPDATES = 1
    G_WARMUP_EPOCHS = 5
    ADV_WARMUP_EPOCHS = 20
    USE_ADA = True
    NPU_DISABLE_R1 = True    # 二阶梯度在 NPU 上不稳定, 禁用 R1
    NPU_DISABLE_R2 = True    # 同理禁用 R2
else:
    NPU_DISABLE_R1 = False
    NPU_DISABLE_R2 = False

# Windows + CUDA: multiprocessing spawn 在 workers>1 时会导致 CUDA 重复初始化,
# NUM_WORKERS=1 使用单 worker 子进程加载, 比 0 (主进程) 更快且不会重复初始化。
import platform
if device_type == "cuda" and platform.system() == "Windows":
    NUM_WORKERS = 1
    print(f"[CONFIG] Windows CUDA: NUM_WORKERS 自动设为 1 (单 worker 加速, 避免 spawn 重复初始化)")
