"""
自适应 OCR 监督 —— 字符级证据权重表

核心思想
  三个 OCR (CaptchaResNet 可能过拟合 / ddddocr 与 ppllocr 为第三方无泄露)
  按字符位给出证据状态, 通过 96 格主表 (C 六态 × D 四态 × P 四态) 计算
  逐位权重; 全部未命中时保留地板权重 W_FLOOR, 避免"无梯度 → 永远读不出"。
  另提供仅 C 可用的快速路径子表 (每步零额外成本) 与三个修正因子:
  M_len(串长≠4) / M_cheat(高置信独苗) / M_cons(D==P 错读共识)。

状态定义 (优先级: hit > near > peer > miss)
  C (CaptchaResNet, 有 logits/置信度):
    c_strong  argmax=label 且 p≥C_CONF_HI
    c_hit     argmax=label 且 C_CONF_LO≤p<C_CONF_HI
    c_weak    argmax=label 且 p<C_CONF_LO
    c_near    argmax≠label 但 (label 为 rank-2 且 p≥C_RANK2_PROB) 或为混淆对
    c_peer    argmax = D 或 P 该位字符 且 ≠label (共识错读)
    c_miss    其他
  D/P (字符串 OCR):
    hit   对齐字符=label
    near  (字符, label) 为实证混淆对
    peer  与另一第三方该位同字 且 ≠label (共识错读)
    miss  其他; 串长≠4 无对齐位置按"无证据"并入 miss (不虚构证据)

权重合成
    w = W_MASTER[c,d,p] × M_len × M_cheat
    若 D==P 整串一致且≠label: w = min(w × M_CONS, M_CONS_CAP)
    最终 w = clamp(w, W_FLOOR, 1.0)
"""
import torch
from config import (
    CHARS, CAPTCHA_LENGTH,
    W_FLOOR, C_CONF_HI, C_CONF_LO, C_RANK2_PROB,
    M_LEN, M_LEN_MIN, M_CHEAT, M_CONS, M_CONS_CAP,
    CONFUSABLE_PAIRS,
)


# ============== OCR 基准准确率 ==============
# 全量实测值 (真实数据 149,886 张, 双 OCR 整图准确率)。
# 欺骗阈值 = 基准 + CHEAT_THRESHOLD, 即"机器识别率明显高于真实数据"时判定欺骗。
OCR_BASELINE = {
    "ddddocr": 0.7983,      # 79.83%
    "ppllocr": 0.6787,      # 67.87%
    "captcha_resnet": 0.90, # 人为降低基准 (该模型在训练集上训练, 易过拟合)
}

# ============== 自适应配置 ==============
CHEAT_THRESHOLD = 0.05     # 超过基准+5%视为欺骗
OCR_WEIGHT_INITIAL = 0.15  # 初始 OCR 外层权重
OCR_WEIGHT_MIN = 0.02      # 最低外层权重
OCR_WEIGHT_DECAY = 0.7     # 检测到欺骗时的衰减因子
OCR_WEIGHT_RECOVER = 1.02  # 未检测到欺骗时的恢复因子

# ============== 状态编码 ==============
C_STRONG, C_HIT, C_WEAK, C_NEAR, C_PEER, C_MISS = \
    "c_strong", "c_hit", "c_weak", "c_near", "c_peer", "c_miss"
C_STATES = (C_STRONG, C_HIT, C_WEAK, C_NEAR, C_PEER, C_MISS)

DP_HIT, DP_NEAR, DP_PEER, DP_MISS = "hit", "near", "peer", "miss"
DP_STATES = (DP_HIT, DP_NEAR, DP_PEER, DP_MISS)

C_SYMBOL = {C_STRONG: "C++", C_HIT: "C+", C_WEAK: "C±",
            C_NEAR: "C~", C_PEER: "C!", C_MISS: "C-"}
DP_SYMBOL = {DP_HIT: "+", DP_NEAR: "~", DP_PEER: "!", DP_MISS: "-"}

# ============== 96 格主表 W_MASTER[c_state][d_state][p_state] ==============
# 合成原则:
#   s = e_C×corr + e_D + e_P (corr=1.0 有第三方证据 / 0.3 C 独苗)
#   7 档塑形 + D!P! 双共识错读 +0.08 (可读性最强反证)
#   C 独苗列 (-,-) 为欺骗嫌疑低权列 (0.20/0.16/0.12/0.08)
W_MASTER = {
    C_STRONG: {
        DP_HIT:  {DP_HIT: 1.00, DP_NEAR: 0.90, DP_PEER: 0.95, DP_MISS: 0.78},
        DP_NEAR: {DP_HIT: 0.90, DP_NEAR: 0.78, DP_PEER: 0.85, DP_MISS: 0.63},
        DP_PEER: {DP_HIT: 0.95, DP_NEAR: 0.85, DP_PEER: 0.98, DP_MISS: 0.70},
        DP_MISS: {DP_HIT: 0.78, DP_NEAR: 0.63, DP_PEER: 0.70, DP_MISS: 0.20},
    },
    C_HIT: {
        DP_HIT:  {DP_HIT: 0.95, DP_NEAR: 0.85, DP_PEER: 0.90, DP_MISS: 0.70},
        DP_NEAR: {DP_HIT: 0.85, DP_NEAR: 0.70, DP_PEER: 0.78, DP_MISS: 0.55},
        DP_PEER: {DP_HIT: 0.90, DP_NEAR: 0.78, DP_PEER: 0.93, DP_MISS: 0.63},
        DP_MISS: {DP_HIT: 0.70, DP_NEAR: 0.55, DP_PEER: 0.63, DP_MISS: 0.16},
    },
    C_WEAK: {
        DP_HIT:  {DP_HIT: 0.90, DP_NEAR: 0.78, DP_PEER: 0.85, DP_MISS: 0.63},
        DP_NEAR: {DP_HIT: 0.78, DP_NEAR: 0.63, DP_PEER: 0.70, DP_MISS: 0.48},
        DP_PEER: {DP_HIT: 0.85, DP_NEAR: 0.70, DP_PEER: 0.86, DP_MISS: 0.55},
        DP_MISS: {DP_HIT: 0.63, DP_NEAR: 0.48, DP_PEER: 0.55, DP_MISS: 0.12},
    },
    C_NEAR: {
        DP_HIT:  {DP_HIT: 0.85, DP_NEAR: 0.70, DP_PEER: 0.78, DP_MISS: 0.55},
        DP_NEAR: {DP_HIT: 0.70, DP_NEAR: 0.55, DP_PEER: 0.63, DP_MISS: 0.40},
        DP_PEER: {DP_HIT: 0.78, DP_NEAR: 0.63, DP_PEER: 0.78, DP_MISS: 0.48},
        DP_MISS: {DP_HIT: 0.55, DP_NEAR: 0.40, DP_PEER: 0.48, DP_MISS: W_FLOOR},
    },
    C_PEER: {
        DP_HIT:  {DP_HIT: 0.88, DP_NEAR: 0.74, DP_PEER: 0.81, DP_MISS: 0.59},
        DP_NEAR: {DP_HIT: 0.74, DP_NEAR: 0.59, DP_PEER: 0.66, DP_MISS: 0.44},
        DP_PEER: {DP_HIT: 0.81, DP_NEAR: 0.66, DP_PEER: 0.82, DP_MISS: 0.51},
        DP_MISS: {DP_HIT: 0.59, DP_NEAR: 0.44, DP_PEER: 0.51, DP_MISS: W_FLOOR},
    },
    C_MISS: {
        DP_HIT:  {DP_HIT: 0.78, DP_NEAR: 0.63, DP_PEER: 0.70, DP_MISS: 0.48},
        DP_NEAR: {DP_HIT: 0.63, DP_NEAR: 0.48, DP_PEER: 0.55, DP_MISS: 0.29},
        DP_PEER: {DP_HIT: 0.70, DP_NEAR: 0.55, DP_PEER: 0.71, DP_MISS: 0.40},
        DP_MISS: {DP_HIT: 0.48, DP_NEAR: 0.29, DP_PEER: 0.40, DP_MISS: W_FLOOR},
    },
}

# ============== 快速路径子表 (仅 C 可用, 即主表 D- P- 列) ==============
W_FAST = {
    C_STRONG: 0.20,
    C_HIT: 0.16,
    C_WEAK: 0.12,
    C_NEAR: W_FLOOR,
    C_PEER: W_FLOOR,   # 无 D/P 无法判定 peer, 不虚构证据
    C_MISS: W_FLOOR,
}

# ============== 混淆对 (双向对称) ==============
_CONFUSABLE = set()
for _a, _b in CONFUSABLE_PAIRS:
    _CONFUSABLE.add((_a, _b))
    _CONFUSABLE.add((_b, _a))


def _is_confusable(a, b):
    return a != b and (a, b) in _CONFUSABLE


def build_confusion_matrix(device=None):
    """36×36 布尔矩阵: CONF_MAT[i,j]=CHARS[i] 与 CHARS[j] 互为混淆对 (i≠j)"""
    n = len(CHARS)
    mat = torch.zeros(n, n, dtype=torch.bool, device=device)
    for i, a in enumerate(CHARS):
        for j, b in enumerate(CHARS):
            if _is_confusable(a, b):
                mat[i, j] = True
    return mat


def align_pred(pred: str, label: str):
    """
    字符串 OCR 输出对齐到标签位:
    串长==4 时直接位对齐; 否则在偏移 o∈[-2..2] 内取匹配数最多的循环对齐。
    对不上的位置记 None (丢失=无证据)。
    Returns: (chars: list[str|None] 长度=CAPTCHA_LENGTH, len_ok: bool)
    """
    n = len(label)
    if len(pred) == n:
        return [pred[i] for i in range(n)], True
    if not pred:
        return [None] * n, False
    best_off, best_score = 0, -1.0
    for o in range(-2, 3):
        score = 0
        for i in range(n):
            j = i + o
            if 0 <= j < len(pred) and pred[j] == label[i]:
                score += 1
        score -= 0.01 * abs(o)   # 平分偏好小偏移
        if score > best_score:
            best_score, best_off = score, o
    chars = []
    for i in range(n):
        j = i + best_off
        chars.append(pred[j] if 0 <= j < len(pred) else None)
    return chars, False


def dp_state(my_char, label_char, peer_char):
    """单个第三方 OCR 的字符位状态. 优先级: hit > near > peer > miss"""
    if my_char is None:
        return DP_MISS          # 丢失=无证据, 不虚构
    if my_char == label_char:
        return DP_HIT
    if _is_confusable(my_char, label_char):
        return DP_NEAR
    if peer_char is not None and my_char == peer_char and my_char != label_char:
        return DP_PEER
    return DP_MISS


def c_state_from_pred(pred_char, label_char, conf, label_rank, p_true,
                      d_char=None, p_char=None):
    """
    C 的字符位状态 (标量版, 供离线分析/测试)。
    label_rank: label 概率在 36 类中的排名 (1=最高)。
    """
    if pred_char == label_char:
        if conf >= C_CONF_HI:
            return C_STRONG
        if conf >= C_CONF_LO:
            return C_HIT
        return C_WEAK
    if _is_confusable(pred_char, label_char):
        return C_NEAR
    if label_rank == 2 and p_true >= C_RANK2_PROB:
        return C_NEAR
    for peer in (d_char, p_char):
        if peer is not None and pred_char == peer and pred_char != label_char:
            return C_PEER
    return C_MISS


def lookup_weight(c, d, p):
    return W_MASTER[c][d][p]


def apply_multipliers(w, d_state, p_state, c_state, conf,
                      d_len_ok, p_len_ok, consensus_wrong,
                      phase_c=False, d_garbage=False, p_garbage=False):
    """
    样本级/位级修正因子:
      M_len  : D 或 P 串长≠4 → ×0.85/个, 叠加下限 M_LEN_MIN
      M_cheat: C 强命中且 p 高, 而 D/P 均未命中 → ×0.50 (高置信独苗=对抗签名)
               Phase C 混淆阶段: 混淆对内 miss 是目标态, 仅越界垃圾预测
               (读成 36 类外字符) 才视为欺骗签名
      M_cons : D==P 整串一致且≠label → ×1.10, 终值 cap M_CONS_CAP
    """
    m = 1.0
    if not d_len_ok:
        m *= M_LEN
    if not p_len_ok:
        m *= M_LEN
    m = max(m, M_LEN_MIN)
    w *= m

    cheat_cond = (d_garbage and p_garbage) if phase_c else (d_state == DP_MISS and p_state == DP_MISS)
    if (c_state in (C_STRONG, C_HIT) and conf >= C_CONF_HI
            and cheat_cond):
        w *= M_CHEAT

    if consensus_wrong:
        w = min(w * M_CONS, M_CONS_CAP)

    return max(min(w, 1.0), W_FLOOR)


class MultiOCRAdaptive:
    """字符级证据权重表 + 自适应外层权重 λ"""

    def __init__(self, device, load_third_party=True):
        """
        Args:
            device: 设备
            load_third_party: 课程学习 Phase A 时可设 False 仅加载 CaptchaResNet
                (节省第三方 OCR 的 CPU 评估开销), 之后调用 load_third_party()
                幂等补载。
        """
        self.device = device

        # 加载 OCR 模型
        self.ocr_models = {}
        self._third_party_loaded = False
        self._load_captcha_resnet()

        self.base_accuracies = OCR_BASELINE.copy()
        self.cheat_threshold = CHEAT_THRESHOLD
        self.current_weight = OCR_WEIGHT_INITIAL

        # 统计
        self.cheat_counts = {name: 0 for name in self.ocr_models}
        self.total_checks = 0

        # 历史记录
        self.weight_history = [OCR_WEIGHT_INITIAL]
        self.accuracy_history = {name: [] for name in self.ocr_models}

        # 状态组合统计 (全表样本) + 快速路径 C 态统计
        self.combo_counts = {}
        self.fast_c_counts = {}
        self.total_samples = 0
        self.total_fast = 0
        self.w_sum = 0.0
        self.valid_sum = 0.0

        # 36×36 混淆矩阵 (C 状态判定用)
        self._conf_mat = build_confusion_matrix(device=device)

        if load_third_party:
            self.load_third_party()

    def _load_captcha_resnet(self):
        """加载 CaptchaResNet (项目内置, OCR 监督 + C 状态判定)"""
        try:
            import ocr_net
            captcha_resnet = ocr_net.load_ocr(device=self.device, freeze=True)
            self.ocr_models['captcha_resnet'] = captcha_resnet
            print(f"  [OCR] 加载 CaptchaResNet 成功")
        except Exception as e:
            print(f"  [OCR] 加载 CaptchaResNet 失败: {e}")

    def load_third_party(self):
        """惰性加载 ddddocr + ppllocr (幂等)"""
        if self._third_party_loaded:
            return

        try:
            import ddddocr
            self.ocr_models['ddddocr'] = ddddocr.DdddOcr(show_ad=False)
            print(f"  [OCR] 加载 ddddocr 成功")
        except Exception as e:
            print(f"  [OCR] 加载 ddddocr 失败: {e}")

        try:
            from ppllocr import OCR
            self.ocr_models['ppllocr'] = OCR()
            print(f"  [OCR] 加载 ppllocr 成功")
        except Exception as e:
            print(f"  [OCR] 加载 ppllocr 失败: {e}")

        self._third_party_loaded = True
        # 补齐统计字典 (延迟加载的模型)
        for name in self.ocr_models:
            self.cheat_counts.setdefault(name, 0)
            self.accuracy_history.setdefault(name, [])
        print(f"  [OCR] 总共加载 {len(self.ocr_models)} 个OCR模型")

    # ─────────────────────────────────────────────────────
    #  字符级权重计算
    # ─────────────────────────────────────────────────────
    @torch.no_grad()
    def compute_char_weights(self, generated_images, labels,
                             ocr_logits=None, max_samples=None,
                             phase_c=False):
        """
        计算每个生成样本每个字符位的证据权重。

        Args:
            generated_images: (B,1,H,W) tanh[-1,1]
            labels: (B,CAPTCHA_LENGTH) 类索引
            ocr_logits: 可选, CaptchaResNet logits (B,4,36), 避免重复前向
            max_samples: None=全部走三 OCR 全表; 0=全部仅快速路径 (C only);
                         N>0=随机 N 张走全表, 其余快速路径
            phase_c: 机器干扰阶段 — 混淆对内 NEAR 视为目标态 (映射为 HIT),
                     M_cheat 仅罚越界垃圾预测

        Returns:
            char_weights: (B, CAPTCHA_LENGTH) tensor on device
            info: dict(w_mean, valid_ratio, n_full, n_fast)
        """
        from config import OCR_VALID_THR, NUM_CLASSES
        B = generated_images.size(0)
        dev = generated_images.device

        # ── 1. C 状态 (每步都算, GPU 向量化) ──
        cap = self.ocr_models.get('captcha_resnet')
        if ocr_logits is None:
            ocr_logits = cap(generated_images)
        logits = ocr_logits.detach().float()
        probs = logits.softmax(-1)                              # (B,4,36)
        pred_idx = probs.argmax(-1)                             # (B,4)
        conf = probs.max(-1).values                             # (B,4)
        p_true = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        # label 概率排名 (1=最高)
        label_rank = (probs > p_true.unsqueeze(-1)).sum(-1) + 1  # (B,4)
        near_mat = self._conf_mat[pred_idx, labels]              # 混淆对 (B,4)
        near_mask = near_mat | ((label_rank == 2) & (p_true >= C_RANK2_PROB))
        hit_mask = pred_idx == labels

        label_chars = [[CHARS[int(t)] for t in row] for row in labels.cpu().tolist()]
        pred_chars = [[CHARS[int(t)] for t in row] for row in pred_idx.cpu().tolist()]

        # ── 2. 选取全表样本 ──
        if max_samples is None:
            full_idx = list(range(B))
        elif max_samples <= 0:
            full_idx = []
        else:
            n = min(max_samples, B)
            perm = torch.randperm(B, device=dev)[:n].cpu().tolist()
            full_idx = sorted(perm)
        full_set = set(full_idx)

        # ── 3. 第三方字符串预测 (仅全表样本) ──
        dddd = self.ocr_models.get('ddddocr')
        ppll = self.ocr_models.get('ppllocr')
        d_preds, p_preds = {}, {}
        if full_idx and (dddd is not None or ppll is not None):
            imgs_np = ((generated_images.detach().cpu() * 0.5 + 0.5) * 255).byte().numpy()
            for i in full_idx:
                img_bytes = self._numpy_to_bytes(imgs_np[i])
                if dddd is not None:
                    try:
                        d_preds[i] = dddd.classification(img_bytes).strip().upper()
                    except Exception:
                        d_preds[i] = ""
                if ppll is not None:
                    try:
                        p_preds[i] = ppll.classification(img_bytes).strip().upper()
                    except Exception:
                        p_preds[i] = ""

        # ── 4. 逐位查表 ──
        char_w = torch.zeros(B, CAPTCHA_LENGTH, dtype=torch.float32, device=dev)
        for i in range(B):
            label_str = ''.join(label_chars[i])

            if i in full_set:
                has_d = i in d_preds
                has_p = i in p_preds
                d_raw = d_preds.get(i, "")
                p_raw = p_preds.get(i, "")
                d_chars, d_len_ok = (align_pred(d_raw, label_str) if has_d
                                     else ([None] * CAPTCHA_LENGTH, True))
                p_chars, p_len_ok = (align_pred(p_raw, label_str) if has_p
                                     else ([None] * CAPTCHA_LENGTH, True))
                consensus_wrong = (has_d and has_p and bool(d_raw)
                                   and d_raw == p_raw and d_raw != label_str)
                # 越界垃圾预测判定 (含非 36 类字符 → M_cheat 依据)
                d_garbage = has_d and bool(d_raw) and any(c not in CHARS for c in d_raw)
                p_garbage = has_p and bool(p_raw) and any(c not in CHARS for c in p_raw)
                for j in range(CAPTCHA_LENGTH):
                    lc = label_chars[i][j]
                    pc = pred_chars[i][j]
                    d_state = dp_state(d_chars[j], lc, p_chars[j]) if has_d else DP_MISS
                    p_state = dp_state(p_chars[j], lc, d_chars[j]) if has_p else DP_MISS
                    # C 状态
                    if hit_mask[i, j]:
                        if conf[i, j] >= C_CONF_HI:
                            cs = C_STRONG
                        elif conf[i, j] >= C_CONF_LO:
                            cs = C_HIT
                        else:
                            cs = C_WEAK
                    elif near_mask[i, j]:
                        cs = C_NEAR
                    else:
                        cs = C_MISS
                        for peer in (d_chars[j], p_chars[j]):
                            if peer is not None and pc == peer and pc != lc:
                                cs = C_PEER
                                break
                    if phase_c:
                        # 混淆对内 NEAR = 目标态 (机器干扰想要的结果), 映射为 HIT
                        if cs == C_NEAR:
                            cs = C_HIT
                        if d_state == DP_NEAR:
                            d_state = DP_HIT
                        if p_state == DP_NEAR:
                            p_state = DP_HIT
                    w = lookup_weight(cs, d_state, p_state)
                    w = apply_multipliers(
                        w, d_state, p_state, cs, float(conf[i, j]),
                        d_len_ok, p_len_ok, consensus_wrong,
                        phase_c=phase_c, d_garbage=d_garbage, p_garbage=p_garbage)
                    char_w[i, j] = w
                    key = (C_SYMBOL[cs], DP_SYMBOL[d_state], DP_SYMBOL[p_state])
                    self.combo_counts[key] = self.combo_counts.get(key, 0) + 1
                self.total_samples += 1
            else:
                # 快速路径: 仅 C 状态
                for j in range(CAPTCHA_LENGTH):
                    if hit_mask[i, j]:
                        if conf[i, j] >= C_CONF_HI:
                            cs = C_STRONG
                        elif conf[i, j] >= C_CONF_LO:
                            cs = C_HIT
                        else:
                            cs = C_WEAK
                    elif near_mask[i, j]:
                        cs = C_NEAR
                    else:
                        cs = C_MISS   # 无 D/P 无法判定 peer, 不虚构
                    if phase_c and cs == C_NEAR:
                        cs = C_HIT    # 混淆对内 NEAR=目标态
                    char_w[i, j] = max(W_FAST[cs], W_FLOOR)
                    self.fast_c_counts[C_SYMBOL[cs]] = self.fast_c_counts.get(C_SYMBOL[cs], 0) + 1
                self.total_fast += 1

        w_mean = float(char_w.mean())
        valid_ratio = float((char_w > OCR_VALID_THR).float().mean())
        self.w_sum += w_mean
        self.valid_sum += valid_ratio

        info = {
            "w_mean": w_mean,
            "valid_ratio": valid_ratio,
            "n_full": len(full_idx),
            "n_fast": B - len(full_idx),
        }
        return char_w, info

    # ─────────────────────────────────────────────────────
    #  外层自适应权重 (欺骗检测)
    # ─────────────────────────────────────────────────────
    def check_and_update(self, generated_images, labels):
        """
        检查生成图像的 OCR 准确率, 更新外层权重。

        Returns:
            current_weight, accuracies, is_cheating, cheating_models
        """
        # 第三方 OCR 未加载 (Phase A) 时不做欺骗检测, 权重保持
        if not self._third_party_loaded:
            self.weight_history.append(self.current_weight)
            return self.current_weight, {}, False, []

        accuracies = {}
        cheating_models = []

        for name, ocr_model in self.ocr_models.items():
            base_acc = self.base_accuracies.get(name, 0.5)
            threshold = base_acc + self.cheat_threshold

            current_acc = self._evaluate_ocr_accuracy(name, ocr_model, generated_images, labels)
            accuracies[name] = current_acc
            self.accuracy_history[name].append(current_acc)

            if current_acc > threshold:
                cheating_models.append(name)
                self.cheat_counts[name] += 1

        self.total_checks += 1

        is_cheating = self._determine_cheating(cheating_models)

        if is_cheating:
            self.current_weight = max(
                OCR_WEIGHT_MIN,
                self.current_weight * OCR_WEIGHT_DECAY
            )
        else:
            self.current_weight = min(
                OCR_WEIGHT_INITIAL,
                self.current_weight * OCR_WEIGHT_RECOVER
            )

        self.weight_history.append(self.current_weight)

        return self.current_weight, accuracies, is_cheating, cheating_models

    def _evaluate_ocr_accuracy(self, name, ocr_model, images, labels):
        """评估 OCR 模型在生成图像上的整串准确率"""
        if name == 'captcha_resnet':
            ocr_model.eval()
            with torch.no_grad():
                logits = ocr_model(images)
                preds = logits.argmax(dim=-1)  # (B, 4)
                correct = (preds == labels).all(dim=1).float().mean()
                return correct.item()

        elif name in ('ddddocr', 'ppllocr'):
            correct = 0
            total = 0

            imgs_np = ((images.cpu() * 0.5 + 0.5) * 255).byte().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(min(len(imgs_np), 32)):  # 仅评估前 32 张, 节省时间
                img_bytes = self._numpy_to_bytes(imgs_np[i])
                label_str = ''.join([CHARS[l] for l in labels_np[i]])

                try:
                    pred = ocr_model.classification(img_bytes).strip().upper()
                    if len(pred) == 4 and pred == label_str:
                        correct += 1
                    total += 1
                except Exception:
                    continue

            return correct / total if total > 0 else 0.0

        return 0.0

    def _determine_cheating(self, cheating_models):
        """
        分层欺骗判定:
        - ppllocr + ddddocr 联合检测: 2 个都超过阈值才算欺骗
        - CaptchaResNet 单独检测: 1 个超过阈值即算欺骗
        """
        third_party_cheat = []
        captcha_resnet_cheat = []

        for name in cheating_models:
            if name in ['ddddocr', 'ppllocr']:
                third_party_cheat.append(name)
            elif name == 'captcha_resnet':
                captcha_resnet_cheat.append(name)

        third_party_is_cheating = len(third_party_cheat) >= 2
        captcha_resnet_is_cheating = len(captcha_resnet_cheat) > 0

        return third_party_is_cheating or captcha_resnet_is_cheating

    def _numpy_to_bytes(self, img_np):
        """将 numpy 图像转换为 JPEG bytes"""
        from PIL import Image
        import io

        img = Image.fromarray(img_np[0])  # (H, W), 单通道灰度
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return buf.getvalue()

    # ─────────────────────────────────────────────────────
    #  状态输出
    # ─────────────────────────────────────────────────────
    def get_status(self):
        """获取当前状态"""
        return {
            "current_weight": self.current_weight,
            "cheat_counts": dict(self.cheat_counts),
            "total_checks": self.total_checks,
            "base_accuracies": dict(self.base_accuracies),
            "cheat_rate": {name: count / max(self.total_checks, 1)
                           for name, count in self.cheat_counts.items()},
            "combo_counts": dict(self.combo_counts),
            "fast_c_counts": dict(self.fast_c_counts),
            "total_samples": self.total_samples,
            "total_fast": self.total_fast,
            "w_mean": self.w_sum / max(self.total_samples + self.total_fast, 1),
            "valid_ratio": self.valid_sum / max(self.total_samples + self.total_fast, 1),
        }

    def get_summary(self):
        """获取一行式摘要 (供训练日志打印)"""
        status = self.get_status()
        parts = [f"OCR权重={status['current_weight']:.4f}"]

        for name in self.ocr_models:
            rate = status['cheat_rate'].get(name, 0)
            base = self.base_accuracies.get(name, 0)
            last_acc = self.accuracy_history[name][-1] if self.accuracy_history[name] else 0
            parts.append(f"{name}={last_acc*100:.1f}%(基准{base*100:.0f}%,欺骗率{rate*100:.1f}%)")

        parts.append(f"w均值={status['w_mean']:.3f},有效率={status['valid_ratio']:.1%}")

        # 状态组合统计 top-4 (全表样本)
        if self.total_samples > 0:
            combo_stats = []
            for combo, count in sorted(self.combo_counts.items(), key=lambda kv: -kv[1])[:4]:
                if count > 0:
                    pct = count / self.total_samples * 100
                    combo_stats.append(f"{combo}:{count}({pct:.1f}%)")
            if combo_stats:
                parts.append(f"组合: {', '.join(combo_stats)}")

        return " | ".join(parts)

    def get_combo_distribution(self):
        """获取组合分布 (含对应权重)"""
        distribution = {}
        total = self.total_samples + self.total_fast
        if total == 0:
            return distribution

        for combo, count in self.combo_counts.items():
            if count > 0:
                c, d, p = combo
                inv_c = {v: k for k, v in C_SYMBOL.items()}
                inv_d = {v: k for k, v in DP_SYMBOL.items()}
                w = lookup_weight(inv_c[c], inv_d[d], inv_d[p])
                distribution[combo] = {
                    'count': count,
                    'percentage': count / total * 100,
                    'weight': w,
                }
        return distribution


if __name__ == "__main__":
    print("OCR 字符级证据权重表")
    print("=" * 50)

    print("\nOCR基准准确率:")
    for name, acc in OCR_BASELINE.items():
        print(f"  {name}: {acc*100:.1f}%")

    print("\n96格主表 (行=D, 列=P):")
    for c in C_STATES:
        print(f"  [{C_SYMBOL[c]}]")
        print(f"    D\\P | {' | '.join(DP_SYMBOL[p] for p in DP_STATES)}")
        for d in DP_STATES:
            row = " | ".join(f"{W_MASTER[c][d][p]:.2f}" for p in DP_STATES)
            print(f"    {DP_SYMBOL[d]:>4} | {row}")

    print("\n快速路径子表 (仅C):")
    for c in C_STATES:
        print(f"  {C_SYMBOL[c]:>4}: {W_FAST[c]:.2f}")
