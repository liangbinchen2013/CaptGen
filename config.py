"""
配置：验证码生成模型训练参数 (v10 架构重写 + 损失均衡版)
所有路径默认相对项目根目录，可通过 CLI 参数 `--data-dir` / `--output-dir` 覆盖

训练规模: 149,888 张, 36 字符 (0-9, A-Z) — 大小写标签不可靠，统一归并为大写
标签说明: 原始标注对大小写随机分配，无法从源头修正。
         将所有标签统一映射为大写 (0-9 + A-Z = 36 类)，消除矛盾监督信号。
         注意: 视觉歧义对已由全量OCR混淆画像实证 (26对, 见 confusion_profile.json);
         v27 另发现标签分布非均匀 (0 缺失, 1/5/O 稀有, I/L/S/U/V 半频)。

v28 核心改动 (2026-09) — 修复 v27 gap 过优化大退步 (回退 v26 视觉基线):
  v27 退步实证 (同口径 48 张 EMA 生成图, ddddocr/ppllocr 整图 acc):
    v26@ep100: dddd 54.2% / ppll 41.7%, 4位长度率 69.8%, gap比 0.77/0.90/0.71 (≈真实)
    v27@ep100: dddd  0.0% / ppll  0.0%, 4位长度率 26.0%, gap比 0.000/0.001/0.000
  根因: v27 gap_ink_loss 为"无目标无下界"的纯最小化; Ep6 开启后 2 轮内把 gap 比
        从 0.5 压到 ~0.002 → 字符间切出硬间隙/笔画截断; CaptchaResNet 四等分
        池化 (ocr_net.py AdaptiveAvgPool2d((1,4))) 反而因此读得更准 (0.993),
        内部指标 (ocr_acc/aux_acc_fake/d_gap) 全"变好", 外部可读性崩塌.
  注: OCR_BASELINE/Phase C (v27) 与本次退步无因果 (Phase B 前不生效).

  - [P0] gap 损失改"上限 hinge": relu(软gap比 - GAP_RATIO_MAX), 低于目标不奖励;
        默认 LAMBDA_GAP=0.0 关闭 (回归 v26 视觉/第三方可读性), --lambda-gap 可开启
  - [P0] 断点续训修复: checkpoint 保存/恢复 LR scheduler 状态;
        旧检查点 (v26 等) 无该字段时按 epoch 快进, 防止续训 LR 跳回初始值
  - [P1] 日志区分"未测量"与"真 0": 第三方 acc 未测记 -1.0, epoch 均值只统计实测
  - [P1] generate.py 默认按真实标签分布采样 (0 缺失, 1/5/O 稀有), --label-dist uniform 可选
  - [P1] evaluate_generated.py: ddddocr use_gpu 自动回退; 输出目录随 OUTPUT_DIR
  - [P1] OUTPUT_DIR → output_v28

v27 核心改动 (2026-09) — 机器干扰画像 + 反粘连 (基于 ocr_error_stats_*.txt 全量实测):
  目标转向: 不是"机器越读越对", 而是"人能读、机器错得像真实数据"。
  真实txt错误画像=机器干扰黄金标准 (ddddocr整图acc 79.8%, ppllocr 67.9%)。
  - [P0] OCR_BASELINE 修正 (adaptive_ocr.py): 0.748/0.629 → 0.7983/0.6787 实测值。
         旧值恰比实测低5pp=CHEAT_THRESHOLD, 欺骗阈值压在真实acc上 → G达到真实级
         可读即被判欺骗(假阳性陷阱), 修正后阈值=真实+5% 自动成为"机器识别太高"守卫
  - [P0] Phase C (Ep300-400) 混淆软目标 CE: y_soft[c]=(1-ε_c)·δ_c + ε_c·confusion_dist[c],
         ε_c=min(error_rate[c], EPS_CAP), 直接复现真实机器混淆画像("机器干扰"),
         逐字符错误率/confusion_dist 来自 confusion_profile.json (双OCR全量合并)
  - [P0] 反粘连 gap 墨迹损失 (Ep6+全程): 3条字符列界带内压笔画 (真实gap/峰值
         =0.84/0.96/0.69 本就粘连, 目标 fake<0.5), 对抗 <LEN> 错误 (真实占27%/28%)
  - [P1] CONFUSABLE_PAIRS 26对实证 (原5对中 2/Z 5/S 8/B 为伪对, 全量混淆画像中不存在)
  - [P1] OCR_VALID_THR 0.30→0.15: 旧阈值高于 W_FAST 上限0.20, 快速路径下恒0 (测量伪影)
  - [P1] CSV: dddd_acc/ppll_acc/gap_ink 列; 里程碑评测 evaluate_generated.py (M1-M8)
  - [P1] OUTPUT_DIR → output_v27

v26 核心改动 (2026-09) — OCR课程学习 (基于v25 263轮人工评测回退诊断):
  根因分析 (v25 不如 v22):
    1. CE 归一化回退: v20用 /w.sum() (只要有样本有权重即全幅), v25用 /(B×4)
       → 有效幅≈w_mean≈13%, 字形梯度弱~8倍
    2. FM 特征匹配被 w_feat 闸门削弱: v20-24 不加权 (λ_eff=0.15×0.6=0.09),
       v25 乘 clamp(w_mean,0.3,1)=0.30 → 0.027 (再弱3倍)
       FM是批级特征分布匹配, 不直接诱导欺骗, 是v20质量的真正功臣
    3. 冷启动死锁: 生成图三OCR全0% → 96格表权重全落地板 → 无可读性梯度

  - [P0] OCR课程学习: Phase A (Ep1-100) 仅CaptchaResNet强引导
         (v20式全幅CE ÷w.sum() + 不加权FM), Ramp (5ep) 线性混合,
         Phase B v25三OCR防欺骗; 切换期2-5ep
  - [P0] 撤销 w_feat 闸门: FM 全程不加权 (OCR_W_FEAT_MIN 废弃)
  - [P1] Phase A 不加载第三方OCR (省每100step的秒级CPU评估, 训练更快)
  - [P1] CSV 新增 ocr_phase/ocr_ramp 列; OUTPUT_DIR → output_v26
  - [P1] curriculum_phase() 为纯epoch函数 → 断点续训自动正确, ckpt无需新字段

v25 核心改动 (2026-09) — Contrast双向化 + OCR字符级证据权重表 + CSV详录:
  基于 v20 output_v20 400轮训练诊断 (training_log.csv 5080行 + 样例分析):
    1. Contrast Loss 恒=0 (5080/5080行, LAMBDA_CONTRAST=3.0完全浪费):
       生成 tanh std≈0.80 ≫ 阈值0.15, 单向下限 relu(0.15-std) 永不触发;
       而真实数据 std≈0.40 (0.16-0.68) — 生成反而"过对比/过噪", 需要双向约束
    2. OCR 监督死锁: 三OCR组合 (F,F,F) 占 99.9%, 8格布尔表全给0权重
       → ocr_ce=0 (日志"OCR: 0.000(有效0%)"), G 无可读性梯度, 永远读不出
    3. CSV 无 ocr_* 列, contrast 恒 0 无告警, 无法事后分析

  - [P0] Contrast 双向: 按真实 batch std 分布带 [q10-margin, q90+margin] 双侧 hinge,
         过高过低都罚 (CONTRAST_MARGIN, 废弃单向下限 CONTRAST_THRESHOLD)
  - [P0] OCR 权重表 v25: 字符级证据权重 (96格主表 + 快速路径 + 3修正因子),
         全未中地板 W_FLOOR=0.08 破死锁; 逐位加权 CE + 固定分母 (B*4),
         防止地板权重被归一化吃掉 (死锁复活)
  - [P1] CSV 详录: ocr_loss/ocr_ce/ocr_feat/ocr_lambda/ocr_w_mean/ocr_valid_ratio/ocr_acc
  - [P1] OUTPUT_DIR → output_v25 (CSV schema 变更, 避免与旧日志混写)

v24 核心改动 (2026-09) — 回滚到v22配置 (v23效果不佳):
  v23问题:
    1. 增强D后训练更不稳定
    2. 需要重新寻找平衡点

  - [P0] 回滚所有v23参数到v22配置

v23 核心改动 (2026-09) — 微调超参数 (基于29条人工反馈分析):
  数据分析:
    1. D太弱: D Gap平均0.13 (应>1.0)，D Loss平均0.94 (接近理论最大1.0)
    2. D(生成)过高: 平均0.27 (应为负值)，D分不清真假
    3. CaptchaResNet过拟合: 25/29条aux_acc_fake=1.0，但人类说"无法识读"
    4. 边缘/细节差: 多条反馈"i模糊"、"7无法识读"、"粘连"

  - [P0] 增强D容量: DIS_FEATURES 50→60，增加判别力
  - [P0] 增加D更新: D_UPDATES 1→2，D需要更强训练
  - [P0] 降低OCR权重: LAMBDA_OCR 0.15→0.10，CaptcharResNet过拟合
  - [P1] 降低对抗压力: ADV_LAMBDA_MAX 0.55→0.50，D太弱时温和对抗
  - [P1] 强化边缘: LAMBDA_EDGE 2.0→2.4，解决"模糊"问题
  - [P1] 强化对比度: LAMBDA_CONTRAST 3.0→3.6，CONTRAST_THRESHOLD 0.15→0.18
  - [P1] 降低OCR CE占比: OCR_CE_WEIGHT 0.4→0.3

v22 核心改动 (2026-09) — 分层OCR检测 (基于v21训练诊断):
  根因分析:
    1. v21移除CaptchaResNet后，OCR监督损失=0，生成质量可能下降
    2. 第三方OCR(ddddocr+ppllocr)只用于欺骗检测，不提供梯度
    3. 需要保留CaptchaResNet的OCR监督，但降低其欺骗检测权重

  分层检测策略:
    1. 第三方OCR联合检测 (ddddocr + ppllocr): 2个都超过阈值才算欺骗，权重高
       - ddddocr: 基准74.8%，阈值79.8%
       - ppllocr: 基准62.9%，阈值67.9%
    2. CaptchaResNet单独检测: 1个超过阈值就算欺骗，权重低
       - 基准90% (人为降低)，阈值95%
       - 额外功能: 提供OCR监督损失 (CE + 特征匹配)

  - [P0] 恢复CaptchaResNet: 提供OCR监督损失，提升生成质量
  - [P0] 分层欺骗判定: 第三方联合检测更可靠，CaptchaResNet权重低
  - [P1] 保留自适应权重: 检测欺骗时降低权重，未欺骗时恢复
  - [P1] 更新OCR基准: ddddocr=74.8%, ppllocr=62.9%, CaptchaResNet=90%

v21 核心改动 (2026-09) — 移除CaptchaResNet，解决数据泄露问题 (基于v20训练诊断):
  根因分析:
    1. CaptchaResNet在训练集上训练，基准99.99%无法作为欺骗检测
    2. 用训练集训练的OCR监督Generator导致数据泄露
    3. Generator学习"欺骗特定OCR"而非"生成人类可读验证码"

  - [P0] 移除CaptchaResNet: 只用ddddocr+ppllocr两个第三方OCR
  - [P0] 更新OCR基准: ddddocr=74.8%, ppllocr=62.9% (真实泛化能力)
  - [P1] 保留自适应权重调整: 检测欺骗时降低权重，未欺骗时恢复
  - [P1] OCR监督损失设为0: 避免数据泄露，只保留欺骗检测

v20 核心改动 (2026-09) — 修复Contrast失效 + 降低OCR基准 (基于v19训练诊断):
  根因分析:
    1. Contrast Loss完全失效: CONTRAST_THRESHOLD=0.20过高，生成图像std>0.20
    2. OCR欺骗仍存在: aux_acc_fake=0.996，接近完全欺骗
    3. CaptchaResNet基准准确率90%仍然太高

  - [P0] 降低Contrast阈值: CONTRAST_THRESHOLD 0.20→0.15，确保Loss生效
  - [P0] 降低OCR基准: CaptchaResNet基准 90%→85%，更容易触发欺骗检测
  - [P1] 保持三OCR自适应监督: ddddocr+ppllocr+CaptchaResNet
  - [P1] 保持Edge Loss: LAMBDA_EDGE=2.0, 动态权重机制
  - [P1] 保持D/G平衡: DIS_FEATURES=50, D_UPDATES=1, ADV_LAMBDA_MAX=0.55

v19 核心改动 (2026-08) — 三OCR自适应监督 (基于v18训练诊断 + OCR预评估):
  根因分析:
    1. v18完全移除OCR监督，导致字形约束丧失，质量反而下降
    2. v17固定OCR权重(0.1)，仍被Generator欺骗
    3. OCR预评估结果: ddddocr=79.8%, ppllocr=67.9%, CaptchaResNet=99%
    4. 需要保留OCR监督，但要防止欺骗

  - [P0] 三OCR自适应监督: 使用ddddocr+ppllocr+CaptchaResNet三个OCR同时监督
         任意一个检测到欺骗就降低权重(0.7x)，未欺骗时恢复(1.02x)
         权重范围: 0.02-0.15
  - [P0] 保持视觉质量: Edge + Contrast + Realism + FM + D Aux
  - [P1] 保持Edge Loss: LAMBDA_EDGE=2.0, 动态权重机制
  - [P1] 保持D/G平衡: DIS_FEATURES=50, D_UPDATES=1, ADV_LAMBDA_MAX=0.55
  - [P1] 保持Cosine学习率: LR_MIN=5e-5

v16 核心改动 (2026-08) — 修复OCR欺骗问题 (基于v15训练诊断):
  根因分析:
    1. OCR Loss过强(LAMBDA_OCR=1.5): Generator只优化"欺骗OCR"而非"人类可读"
    2. aux_acc_fake从Epoch 50就达到0.9996, 但生成图像肉眼不可识读
    3. Contrast Loss完全失效(始终=0.0000): 阈值0.15过低, 生成图像std>0.15
    4. 单一OCR模型容易被对抗样本欺骗

  - [P0] 大幅降低OCR权重: LAMBDA_OCR 1.5→0.3, 防止Generator过度优化OCR
  - [P0] 提高Contrast阈值: CONTRAST_THRESHOLD 0.15→0.25, 真实验证码std≈0.43-0.51
  - [P0] 强化Contrast权重: LAMBDA_CONTRAST 1.0→2.0, 强制生成高对比度图像
  - [P1] 保持Edge Loss: LAMBDA_EDGE=2.0, 维持边缘清晰度
  - [P1] 保持Realism Loss: LAMBDA_REALISM=1.0, 维持分布匹配

v15 核心改动 (2026-08) — 修复v14 D/G失衡 + 学习率过低 (基于v14 400轮诊断):
  根因分析:
    1. D过强: DIS_FEATURES 56 + D_UPDATES 2 + D_ADAPTIVE_THRESHOLD 0.5 三重叠加
       → d_fake_mean从-1.12降到-1.55, G Loss飙升47% (1.53→2.25)
    2. Cosine学习率衰减过深: 最终仅1e-5, G后期完全无法学习
    3. 对抗权重+D增强叠加: ADV_LAMBDA_MAX 0.65 + 更强D = 双重压制G
    4. R1惩罚不稳定: 出现84.77极端值 (v13最高仅0.32)

  - [P0] 恢复D容量: DIS_FEATURES 56→50 (折中v13的48和v14的56)
  - [P0] 恢复D更新频率: D_UPDATES 2→1, 避免D过度压制G
  - [P0] 调整对抗权重: ADV_LAMBDA_MAX 0.65→0.55 (适度提升但不过激)
  - [P0] 调整学习率: LR_MIN 1e-5→5e-5, 匹配v13最终学习率
  - [P1] 恢复R1/R2惩罚: LAMBDA_R1/R2 0.8→0.6 (折中方案)
  - [P1] 保持Edge Loss改进: LAMBDA_EDGE=2.0 + 动态权重机制有效
  - [P1] 保持Contrast改进: LAMBDA_CONTRAST=1.0 + 阈值0.15有效
  - [P1] 延长训练: EPOCHS 400→350 (v14显示400轮后期已无改善)

v14 核心改动 (2026-08) — 提升生成质量 (基于v13 300轮诊断):
  根因分析:
    1. Edge Loss衰减63% (0.19→0.07), 字符边缘监督失效
    2. D判别能力偏弱: d_gap仅2.29, 假数据得分过高(-1.12)
    3. 对抗权重过低(0.5), G缺乏质量提升压力
    4. 阶梯学习率衰减导致后期停滞(5e-5)
    5. R1/R2惩罚波动大, D正则化不稳定

  - [P0] 强化Edge Loss: LAMBDA_EDGE 1.5→2.0, 新增动态权重机制
         Edge Loss低于阈值(0.10)时权重自动翻倍, 防止衰减
  - [P0] 提升对抗权重: ADV_LAMBDA_MAX 0.5→0.65, 增加G的质量压力
  - [P0] 增强D容量: DIS_FEATURES 48→56, 提升判别力到d_gap>3.0
  - [P0] 改用Cosine学习率: USE_COSINE_LR=True, LR_MIN=1e-5
         替代阶梯衰减, 保持后期学习率活力
  - [P1] 强化R1/R2惩罚: LAMBDA_R1/R2 0.5→0.8, 稳定D正则化
  - [P1] 增加D更新频率: D_UPDATES 1→2, D_UPDATES_MAX 3→4
         D_ADAPTIVE_THRESHOLD 0.3→0.5, 更早触发自适应更新
  - [P1] 强化Contrast损失: LAMBDA_CONTRAST 0.8→1.0, CONTRAST_THRESHOLD 0.18→0.15
  - [P1] 延长训练: EPOCHS 300→400, 配合Cosine调度充分训练

v10 核心改动 (2026-08) — 解决过度拟合 + 字符样式缺失 (基于v9 236轮诊断):
  根因分析:
    1. Aux Loss权重(5.0) vs 对抗权重(0.15) = 33:1, Aux主导所有梯度
    2. init_proj单层1250万参数, G轻易记住训练集统计平均
    3. Contrast/Sharpness损失始终为0, 完全失效
    4. 无直接像素/边缘监督, FM权重过低(0.5)
    5. 预热期(Epoch 8)后所有指标停滞228轮, 训练完全浪费

  - [P0] Generator重写: 废除 init_proj (512→24576 单层),
         改为渐进式 CondUpBlock ×3: 4×6 → 8×12 → 16×24 → 35×90,
         每级注入文本条件(FiLM调制), Self-Attention在bottleneck。
         参数量降低60%+。
  - [P0] 损失权重均衡: LAMBDA_AUX_G 5.0→2.0, ADV_LAMBDA_MAX 0.15→1.0,
         LAMBDA_FM 0.5→1.0, 三种信号在G loss中达到约2:1:1平衡。
  - [P0] 新增边缘损失(LAMBDA_EDGE=1.0): Sobel边缘L1匹配,
         直接约束G生成锐利字符笔画, 替代失效的Sharpness损失。
  - [P0] 新增分布匹配(LAMBDA_REALISM=0.5): 像素均值/std/梯度幅度
         三通道统计量匹配, 批量级无配对约束。
  - [P1] 预热缩短: G_WARMUP_EPOCHS 8→5, ADV_WARMUP_EPOCHS 30→15,
         更早引入对抗信号防止预热期局部最优固化。
  - [P1] Contrast阈值调整: 0.35→0.25, 让该损失重新激活。
  - [P1] D简化: 降低通道数(CUDA 56→48, NPU 96→64), 减少D过拟合。

v11 核心改动 (2026-08) — 修复判别器崩溃 + 对抗平衡 (基于v10 300轮诊断):
  根因分析:
    1. D完全崩溃: epoch 300时d_real≈+1.1, d_fake≈+1.2, D无法区分真假
    2. G Loss为负值: D太弱导致G对抗信号反向
    3. D_UPDATES=1不足以在D落后时追赶
    4. ADV_LAMBDA_MAX=1.0对当前G/D平衡过于激进
    5. D容量不足: DIS_FEATURES 64 (NPU) 被v10重写的G碾压
    6. Edge/Realism/Diversity权重过低, 无法弥补对抗信号缺失

  - [P0] D容量恢复: NPU DIS_FEATURES 64→96, 恢复D判别力
  - [P0] 自适应D更新: 基础D_UPDATES=1, 当D Gap<阈值时自动增加到2-3
  - [P0] 对抗权重收敛: ADV_LAMBDA_MAX 1.0→0.5, 温和对抗防止D崩溃
  - [P0] D预训练: D_PRETRAIN_EPOCHS 0→3, 预热前先让D学会区分真假
  - [P1] 质量损失强化: LAMBDA_EDGE 1.0→1.5, LAMBDA_REALISM 0.5→1.0,
         LAMBDA_DIVERSITY 0.5→1.0, 非对抗质量信号占据更大比重
  - [P1] Contrast阈值: 0.25→0.18 (CONTRAST_THRESHOLD), 真正激活
  - [P1] Aux权重微调: LAMBDA_AUX_G 2.0→1.5, 进一步减少Aux霸权
  - [P1] 对抗渐进延长: ADV_WARMUP_EPOCHS 15→20, 更平滑过渡

v9 核心改动 (2026-07) — 修复全局模糊:
  - NOISE_STD 0.06→0.02, ADV_LAMBDA_MAX 0.3→0.05
  - init_h/w 8→16, UpsampleBlock 5→4, nearest+Conv替代bilinear
  - 新增 LAMBDA_SHARPNESS=0.3: Laplacian方差锐度正则化
"""
from pathlib import Path

# ─── 数据 ────────────────────────────────────────────────
DATA_DIR = Path("训练数据")
IMG_HEIGHT = 35
IMG_WIDTH = 90
IMG_CHANNELS = 1  # 灰度图

# 字符集: 仅保留 0-9 + A-Z = 36
# 所有小写标签自动转为大写，消除大小写标签噪声
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARS)  # 36
CAPTCHA_LENGTH = 4  # 验证码长度

# 标签统一映射: 所有字母转为大写
# 由于原始标注大小写随机，舍弃大小写信息是最安全的策略
LABEL_CASE_MAP = {}

# 构建 LABEL_CASE_MAP: a→A, b→B, ..., z→Z
for _c in "abcdefghijklmnopqrstuvwxyz":
    LABEL_CASE_MAP[_c] = _c.upper()

# ─── 训练 ────────────────────────────────────────────────
BATCH_SIZE = 128            # GPU 安全默认, NPU 自动覆盖为 1024
EPOCHS = 400                 # v17: 350→400, 充分利用更多数据(150K)
LEARNING_RATE_G = 2e-4      # GPU 安全默认, NPU 自动覆盖为 5e-4
LEARNING_RATE_D = 2e-4
BETA1 = 0.0             # v4: 0.5→0.0, SNGAN 标准配置, 配合 SpectralNorm + Hinge Loss
BETA2 = 0.9

# ─── 模型 ────────────────────────────────────────────────
LATENT_DIM = 256             # v6: 128→256, 增大噪声空间提升同标签多样性
EMBED_DIM = 64
GEN_FEATURES = 64           # GPU 安全默认, NPU 自动覆盖为 128 (v10: Generator已重写, 独立于GEN_FEATURES)
DIS_FEATURES = 50            # v24: 回滚到v22配置
NOISE_STD = 0.05             # v12: 语义变更 — 现为 CondUpBlock/NoiseInjection 可学习增益的初始值
                             # v9: 0.06→0.02 曾用于固定常数防"抹平"; v12 改为可学习增益+推理也注入,
                             # 配合同标签双z多样性损失根治 z 被忽略 (v11 300轮诊断)

# ─── 损失权重 ────────────────────────────────────────────
LAMBDA_AUX_G = 1.5           # v11: 2.0→1.5, 进一步降低Aux霸权, 让Edge/Realism有更大梯度份额
LAMBDA_AUX_D = 1.0           # 保持, D Aux对G的字符识别监督
LAMBDA_GP  = 10.0            # 仅 CUDA 有效, NPU 用 Hinge Loss
LAMBDA_R1 = 0.6              # v15: 0.8→0.6, 折中避免梯度爆炸
LAMBDA_R2 = 0.6              # v15: 0.8→0.6, R1+R2对称调整
LAMBDA_DIVERSITY = 1.0       # v11: 0.5→1.0, 强化多样性防止同标签输出趋同
ADV_LAMBDA_MAX = 0.55        # v24: 回滚到v22配置

# ─── Aux 分类器稳定性 ───────────────────────────────────
NORMALIZE_AUX = True         # v7: ReACGAN hypersphere 投影 — L2 normalize Aux head 输入特征,
                             # 防止分类器梯度爆炸, 稳定 AC-GAN 训练 (Kang et al., NeurIPS 2021)

# ─── 对抗增强损失权重 ────────────────────────────────────
LAMBDA_FM = 1.0              # v10: 0.5→1.0, FM恢复为主要特征指引, 独立于对抗权重
LAMBDA_CONTRAST = 3.0        # v24: 回滚到v22配置
# v25: 双向对比度匹配 — 真实分布带 [q10-margin, q90+margin] 双侧 hinge
# 生成 std 过低(灰糊)或过高(过噪)都惩罚, 不再只罚过低
# 实测: 真实 tanh std≈0.40 (q10≈0.25, q90≈0.58), v20 生成≈0.80
CONTRAST_MARGIN = 0.05       # v25: 分布带外扩 margin (带外才惩罚)
CONTRAST_LO_Q = 0.10         # v25: 真实 std 分布下分位
CONTRAST_HI_Q = 0.90         # v25: 真实 std 分布上分位
LAMBDA_EDGE = 2.0            # v24: 回滚到v22配置
EDGE_LOSS_MIN = 0.10         # v14: Edge Loss下限, 低于此值时权重翻倍
LAMBDA_REALISM = 1.0         # v11: 0.5→1.0, 分布匹配权重翻倍以弥补对抗信号减弱
LAMBDA_CONSIST = 0.0
LAMBDA_SHARPNESS = 0.0       # v10: 禁用, Laplacian峰度对灰度验证码无效, 被EDGE+REALISM替代

# ─── v25: OCR 字符级证据权重表 (状态机+查表见 adaptive_ocr.py) ─
# 三个OCR (CaptchaResNet/过拟合, ddddocr/ppllocr/第三方无泄露) 按字符位
# 给出 6×4×4=96 格组合权重; 全未中地板 W_FLOOR 破除 v24 死锁
# (F,F,F)→0 → 有效0% → 无梯度 → 永远读不出
LAMBDA_OCR = 0.15            # v25: OCR权重（会被自适应管理器动态调整）
OCR_CE_WEIGHT = 0.4          # v25: CaptchaResNet CE损失占比 (特征匹配占0.6)
OCR_WARMUP_EPOCHS = 5        # 从G预热结束后线性引入

# v25 权重表常量
W_FLOOR = 0.08               # 全未中/丢失位地板: 破死锁但梯度极弱
                             # 外层λ=0.15 时实际强度≈0.012, 防欺骗
C_CONF_HI = 0.90             # C 强命中(C++)置信度阈值
C_CONF_LO = 0.50             # C 命中(C+)/弱命中(C±)分界
C_RANK2_PROB = 0.25          # C~近失: label为rank-2 且概率≥此值
M_LEN = 0.85                 # 串长≠4 结构折损 (D/P各计一次)
M_LEN_MIN = 0.65             # M_LEN 叠加下限
M_CHEAT = 0.50               # "高置信独苗"惩罚: C∈{C++,C+}且p≥0.90且D/P均未命中
M_CONS = 1.10                # D==P 整串一致且≠label: 可读性共识奖励
M_CONS_CAP = 0.95            # 共识奖励后终值上限
# OCR_W_FEAT_MIN = 0.30      # [v26废弃] w_feat闸门: FM是批级分布匹配不诱导欺骗,
                             # v25加闸门反把质量功臣削弱3倍, v26恢复全程不加权
OCR_VALID_THR = 0.15         # v27: 0.30→0.15 (旧值高于 W_FAST 上限0.20, 快速路径恒0伪影)

# ─── v27: 机器混淆画像 (confusion_profile.json, build_confusion_profile.py 生成) ───
# 真实数据 ddddocr+ppllocr 全量 (149,886图) 错误画像:
#   error_rate[c]        逐字符真实错误率 (I=0.37 L=0.34 O=0.35 J=0.17 ...)
#   confusion_dist[c][p] 有向混淆分布 (软目标用, 仅36类内归一化)
#   confusable_pairs     26对强证据混淆 (双向各>=100且合计>=500, 或单向双OCR各>=150)
# 半信用混淆对 (v27: 5对人工猜测 → 26对全量实证; 2/Z 5/S 8/B 被证伪剔除)
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
    CONFUSABLE_PAIRS = (("0", "O"), ("1", "I"), ("2", "Z"), ("5", "S"), ("8", "B"))  # fallback

# Phase C 软目标原料 (缺失时 Phase C 自动退化为硬 CE)
CONFUSION_DIST = (_confusion_profile or {}).get("confusion_dist", {})
OCR_ERROR_RATE = (_confusion_profile or {}).get("error_rate", {})

# ─── v27: Phase C (Ep300-400) 混淆软目标 CE ───────────────────
# y_soft[c] = (1-ε_c)·δ_c + ε_c·confusion_dist[c], ε_c = min(error_rate[c], EPS_CAP)
# 目的: G 画的 I 让 OCR "七成读 I, 其余读 L/T/1…" = 复现真实机器混淆 (机器干扰)
PHASE_C_EPOCHS = 300         # Phase C 起始轮 (--phase-c-epoch 可覆盖)
PHASE_C_RAMP = 5             # ε 线性爬升轮数 (Ep300→305: 0→ε_c)
EPS_CAP = 0.35               # ε 上限: 防软目标过散丧失可读性 (可读性守卫 M8)

# ─── v27/v28: 反粘连 gap 墨迹损失 (可选, 默认关闭) ─────────────
# 字符列界带 (CaptchaResNet 四等分池化列界 x=22.5/45/67.5 附近) 内压笔画对比度.
# 真实数据 gap墨迹/峰值 = 0.84/0.96/0.69 (本就粘连).
# v27 教训: 无下界最小化把 gap 比从 0.5 压到 ~0.002 → 硬间隙/断笔, 第三方OCR 0%/0%.
# v28 修复: 仅当软 gap 比 > GAP_RATIO_MAX 时惩罚 relu(ratio-target), 低处不奖励;
#           默认权重 0 = 关闭 (保持 v26 视觉), 实验时用 --lambda-gap 开启.
LAMBDA_GAP = 0.0             # v28: 0=关闭 (v26行为); >0 时启用上限 hinge
GAP_RATIO_MAX = 0.5          # v28: hinge 上限 — 软gap比超过此值才惩罚
GAP_STRIPS = ((21, 25), (43, 47), (66, 70))   # 3条列界带 [x0,x1)
GAP_INK_TAU = 0.15           # 墨迹判定: |灰度-背景| > tau 计为笔画

# ─── v26: OCR 课程学习 (Curriculum OCR) ────────────────────
# Phase A (Ep 1-CURRICULUM_EPOCHS): 仅 CaptchaResNet 强引导
#   - 第三方OCR不加载 (省CPU), 不做欺骗检测
#   - OCR CE 全幅 (v20式 /w.sum() 归一化), FM 不加权 → 拉起可读性
#   - 允许 aux_acc_fake 偏高 (阶段性接受欺骗, v20教训: 质量优先)
# Ramp (随后 CURRICULUM_RAMP ep): A→B 线性混合 (2-5ep)
# Phase B (之后): v25 三OCR 防欺骗 (96格主表 + 快速路径 + 3修正因子)
USE_CURRICULUM = True
CURRICULUM_EPOCHS = 100      # Phase A 轮数 (--curriculum-epochs 可覆盖)
CURRICULUM_RAMP = 5          # 切换期轮数 (--curriculum-ramp 可覆盖, 2-5)


def curriculum_phase(epoch: int, curriculum_epochs: int = None,
                     curriculum_ramp: int = None, enabled: bool = None):
    """
    v26: 纯 epoch 函数 → ('A'|'AB'|'B', t), t∈[0,1] 为向 B 的混合系数.
    断点续训自动正确 (阶段只由 epoch 决定, checkpoint 无需新字段).

    Returns:
        phase: 'A'=ResNet强引导, 'AB'=切换期, 'B'=v25防欺骗
        t: 0.0=纯A, 1.0=纯B, 中间值=A/B线性混合比例
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
    v27: Phase C 软目标 ε 爬升系数 (纯 epoch 函数, 断点续训自动正确).
    Returns: (in_phase_c: bool, t: float∈[0,1])
      t=0 → 硬CE; t=1 → 全幅 ε_c; Ep300-305 线性爬升
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
D_UPDATES = 1                # v24: 回滚到v22配置
D_UPDATES_MAX = 3            # v24: 回滚到v22配置
D_ADAPTIVE_THRESHOLD = 0.3   # v15: 0.5→0.3, 恢复v13阈值
D_PRETRAIN_EPOCHS = 3        # v11: 0→3, G预热前先训练D学会区分真假, 建立baseline判别力
G_WARMUP_EPOCHS = 5          # v10: 8→5, 缩短纯Aux预热, 更早引入对抗防止局部最优固化
ADV_WARMUP_EPOCHS = 20       # v11: 15→20, 配合更低ADV_LAMBDA_MAX, 更平滑过渡到联合训练
CLIP_NORM = 10.0

# ─── EMA ──────────────────────────────────────────────────
EMA_DECAY = 0.999            # v4: 0.995→0.999, 恢复原窗口 1000 步, 更稳定的推理权重

# ─── 学习率调度 ────────────────────────────────────────
LR_MILESTONES = [150, 250]   # 保留用于MultiStepLR备选
LR_DECAY_FACTOR = 0.5
USE_COSINE_LR = True         # v14: 启用Cosine Annealing替代阶梯衰减
LR_MIN = 5e-5                # v24: 回滚到v22配置

# ─── ADA (自适应判别器增强, StyleGAN2-ADA) ────────────────
USE_ADA = True               # v8: 新增, 对D输入应用随机增强防止过拟合
ADA_TARGET = 0.6             # v8: D对真实数据正确率目标 (r_t), 越高→增强越强
ADA_INTERVAL = 4             # v8: 每4个step调整一次增强概率p
ADA_P_INC = 2e-4             # v8: 每次调整p的增量
ADA_P_DEC = 5e-5             # v8: 每次调整p的减量 (降低更慢, 保守策略)
ADA_AUGMENT_PIPE = [         # v8: 增强pipeline (与StyleGAN2-ADA一致)
    "blit",                  # 像素级: 翻转/旋转/缩放/各向异性
    "geom",                  # 几何: 平移/拉伸 (不泄露字符位置信息)
    "color",                 # 颜色: 亮度/对比度/色调
    "noise",                 # 噪声: 高斯噪声
]

# ─── 标签平滑 ───────────────────────────────────────────
LABEL_SMOOTHING_EPS = 0.03

# ─── 保存 ────────────────────────────────────────────────
OUTPUT_DIR = Path("output_v28")  # v28: v26核心 + gap修复 (gap默认关闭 + 续训scheduler)
SAMPLE_INTERVAL = 1
MODEL_SAVE_INTERVAL = 10
LOG_INTERVAL = 20

# ─── 硬件 ────────────────────────────────────────────────
NUM_WORKERS = 4

# ═══════════════════════════════════════════════════════════
#  设备自适应 (必须在其他模块导入前执行)
# ═══════════════════════════════════════════════════════════
# device_utils 不依赖 config, 无循环导入风险
try:
    from device_utils import device_type
except ImportError:
    device_type = "cpu"

if device_type == "npu":
    # Ascend 910B3 90GB HBM
    BATCH_SIZE = 1024
    GEN_FEATURES = 128
    DIS_FEATURES = 96        # v11: 64→96, 恢复D容量防止被v10重写的G碾压
    LEARNING_RATE_G = 5e-4
    LEARNING_RATE_D = 5e-4
    EPOCHS = 300
    NUM_WORKERS = 0
    LOG_INTERVAL = 20
    LR_MILESTONES = [150, 250]
    D_UPDATES = 1            # v11: 基础保持1, 自适应逻辑按需增加到2-3
    G_WARMUP_EPOCHS = 5
    ADV_WARMUP_EPOCHS = 20   # v11: 15→20, 平滑过渡
    USE_ADA = True
    NPU_DISABLE_R1 = True
    NPU_DISABLE_R2 = True
else:
    NPU_DISABLE_R1 = False
    NPU_DISABLE_R2 = False

# Windows + CUDA: multiprocessing spawn 在 workers>1 时会导致 CUDA 重复初始化
# NUM_WORKERS=1 使用单 worker 子进程加载, 比 0 (主进程) 更快且不会重复初始化
import platform
if device_type == "cuda" and platform.system() == "Windows":
    NUM_WORKERS = 1
    print(f"[CONFIG] Windows CUDA: NUM_WORKERS 自动设为 1 (单 worker 加速, 避免 spawn 重复初始化)")
