"""OCR 字符级证据权重表 / 课程学习 / 软目标 / gap 损失 单元测试 (纯CPU, 不加载模型)"""
import torch
from adaptive_ocr import (
    W_MASTER, W_FAST, C_STATES, DP_STATES, C_SYMBOL, DP_SYMBOL,
    align_pred, dp_state, c_state_from_pred, lookup_weight, apply_multipliers,
    _is_confusable,
)
from config import (W_FLOOR, M_LEN, M_LEN_MIN, M_CHEAT, M_CONS, M_CONS_CAP,
                    CHARS, NUM_CLASSES, GAP_STRIPS, EPS_CAP)

fails = []

def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        fails.append(name)
        print(f"  FAIL {name} {detail}")

print("== 1. 主表结构完整性 (6x4x4=96格) ==")
n = 0
for c in C_STATES:
    for d in DP_STATES:
        for p in DP_STATES:
            w = lookup_weight(c, d, p)
            n += 1
            check(f"W[{C_SYMBOL[c]},{DP_SYMBOL[d]},{DP_SYMBOL[p]}]={w:.2f}",
                  0 < w <= 1.0, f"越界: {w}")
check("96格齐全", n == 96, f"实际{n}")

print("== 2. 地板/上限 ==")
check("全未中=地板", lookup_weight("c_miss", "miss", "miss") == W_FLOOR)
check("C独苗强命中=0.20", lookup_weight("c_strong", "miss", "miss") == 0.20)
check("三中=1.00", lookup_weight("c_strong", "hit", "hit") == 1.00)
check("双共识错读 C++ !!=0.98", lookup_weight("c_strong", "peer", "peer") == 0.98)
check("快速路径 C++=0.20", W_FAST["c_strong"] == 0.20)
check("快速路径 全部>=地板", all(W_FAST[c] >= W_FLOOR for c in C_STATES))

print("== 3. 单调性: 证据越多权重越高 ==")
for c in C_STATES:
    check(f"{C_SYMBOL[c]} hit,hit > hit,miss",
          lookup_weight(c, "hit", "hit") > lookup_weight(c, "hit", "miss"))
    check(f"{C_SYMBOL[c]} hit,hit > miss,miss",
          lookup_weight(c, "hit", "hit") > lookup_weight(c, "miss", "miss"))
    check(f"{C_SYMBOL[c]} peer,peer >= peer,miss",
          lookup_weight(c, "peer", "peer") >= lookup_weight(c, "peer", "miss"))
check("C++ D+P- > C- D+P-", lookup_weight("c_strong", "hit", "miss") > lookup_weight("c_miss", "hit", "miss"))
check("C独苗列 c_strong>c_hit>c_weak>c_miss",
      lookup_weight("c_strong", "miss", "miss") > lookup_weight("c_hit", "miss", "miss")
      > lookup_weight("c_weak", "miss", "miss") >= lookup_weight("c_miss", "miss", "miss"))

print("== 4. 混淆对 (26对全量实证) ==")
check("0/O", _is_confusable("0", "O") and _is_confusable("O", "0"))
check("I/L", _is_confusable("I", "L") and _is_confusable("L", "I"))
check("1/L", _is_confusable("1", "L"))
check("9/G", _is_confusable("9", "G"))
check("自身不算", not _is_confusable("0", "0"))
check("5/S不在实证对", not _is_confusable("5", "S"))
check("2/Z不在实证对", not _is_confusable("2", "Z"))
check("8/B不在实证对", not _is_confusable("8", "B"))
check("6/G不在实证对", not _is_confusable("6", "G"))

print("== 5. 对齐 ==")
chars, ok = align_pred("ABCD", "ABCD")
check("len4对齐", ok and chars == ["A", "B", "C", "D"])
chars, ok = align_pred("ABC", "ABCD")
check("len3无len_ok", not ok)
check("len3对齐前3", chars[:3] == ["A", "B", "C"] and chars[3] is None)
chars, ok = align_pred("", "ABCD")
check("空串全丢", chars == [None]*4 and not ok)
chars, ok = align_pred("XABCD", "ABCD")
check("len5右移对齐", not ok and "A" in chars)

print("== 6. D/P 状态优先级 hit > near > peer > miss ==")
check("hit", dp_state("A", "A", "B") == "hit")
check("near(混淆)", dp_state("0", "O", "B") == "near")
check("near优先于peer", dp_state("0", "O", "0") == "near")  # 与peer同字但是混淆对
check("peer", dp_state("X", "A", "X") == "peer")
check("miss", dp_state("X", "A", "Y") == "miss")
check("丢失误", dp_state(None, "A", "B") == "miss")

print("== 7. C 状态 ==")
check("strong", c_state_from_pred("A", "A", 0.95, 1, 0.95) == "c_strong")
check("hit", c_state_from_pred("A", "A", 0.7, 1, 0.7) == "c_hit")
check("weak", c_state_from_pred("A", "A", 0.3, 1, 0.3) == "c_weak")
check("near混淆", c_state_from_pred("0", "O", 0.9, 1, 0.1) == "c_near")
check("near rank2", c_state_from_pred("X", "A", 0.5, 2, 0.3) == "c_near")
check("peer", c_state_from_pred("X", "A", 0.5, 5, 0.05, d_char="X") == "c_peer")
check("miss", c_state_from_pred("X", "A", 0.5, 5, 0.05) == "c_miss")

print("== 8. 修正因子 ==")
w = apply_multipliers(0.20, "miss", "miss", "c_strong", 0.95, True, True, False)
check(f"M_cheat独苗 0.20*0.5=0.10 (实际{w:.3f})", abs(w - 0.10) < 1e-6)
w = apply_multipliers(0.50, "hit", "hit", "c_strong", 0.95, False, True, False)
check(f"M_len单侧 0.50*0.85=0.425 (实际{w:.3f})", abs(w - 0.425) < 1e-6)
w = apply_multipliers(0.50, "hit", "hit", "c_strong", 0.95, True, False, True)
check(f"M_cons 0.50*0.85*1.1=0.4675 (实际{w:.3f})", abs(w - 0.4675) < 1e-6)
w = apply_multipliers(0.90, "peer", "peer", "c_strong", 0.95, True, True, True)
check(f"M_cons终值cap 0.90*1.1→{M_CONS_CAP} (实际{w:.3f})", abs(w - M_CONS_CAP) < 1e-6)
w = apply_multipliers(0.20, "miss", "miss", "c_miss", 0.1, False, False, False)
expect = 0.20 * max(M_LEN * M_LEN, M_LEN_MIN)
check(f"M_len双侧叠加下限 0.20*{max(M_LEN*M_LEN, M_LEN_MIN):.4f}={expect:.4f} (实际{w:.4f})",
      abs(w - expect) < 1e-6)
w = apply_multipliers(0.05, "miss", "miss", "c_miss", 0.1, True, True, False)
check(f"地板托底 (实际{w:.3f})", w >= W_FLOOR)
w = apply_multipliers(0.98, "peer", "peer", "c_strong", 0.95, True, True, True)
check(f"M_cons最高不越cap (实际{w:.3f})", w <= M_CONS_CAP + 1e-9)

print("== 9. curriculum_phase 端点 ==")
from config import curriculum_phase
check("Ep1 → A t=0", curriculum_phase(1, 100, 5) == ("A", 0.0))
check("Ep100 → A t=0 (末轮仍A)", curriculum_phase(100, 100, 5) == ("A", 0.0))
ph, t = curriculum_phase(101, 100, 5)
check(f"Ep101 → AB t=0.2 (实际{t})", ph == "AB" and abs(t - 0.2) < 1e-9)
ph, t = curriculum_phase(103, 100, 5)
check(f"Ep103 → AB t=0.6 (实际{t})", ph == "AB" and abs(t - 0.6) < 1e-9)
ph, t = curriculum_phase(105, 100, 5)
check(f"Ep105 → B t=1 (实际{ph},{t})", ph == "B" and t == 1.0)
check("Ep106 → B t=1", curriculum_phase(106, 100, 5) == ("B", 1.0))
check("禁用 → B t=1", curriculum_phase(50, 100, 5, enabled=False) == ("B", 1.0))
check("ramp=0 Ep101瞬切B", curriculum_phase(101, 100, 0) == ("B", 1.0))
ph, t = curriculum_phase(102, 100, 2)
check(f"ramp=2 Ep102 → B t=1 (实际{ph},{t})", ph == "B" and t == 1.0)
ph, t = curriculum_phase(101, 100, 2)
check(f"ramp=2 Ep101 → AB t=0.5 (实际{t})", ph == "AB" and abs(t - 0.5) < 1e-9)
check("Ep400 → B", curriculum_phase(400, 100, 5) == ("B", 1.0))

print("== 10. Ramp 混合数学端点 ==")
# 模拟 ramp 混合公式: w=(1-t)wA+t*wB, ce=(1-t)ceA+t*ceB
import torch as _torch
ce_char = _torch.tensor([[2.0, 0.0, 1.0, 3.0]])
w_fast = _torch.tensor([[0.08, 0.20, 0.12, 0.08]])
w_full = _torch.tensor([[0.48, 0.95, 0.70, 0.08]])
B4 = 4.0
for t_val, name in [(0.0, "t=0→纯A"), (0.5, "t=0.5→混合"), (1.0, "t=1→纯B")]:
    w = (1 - t_val) * w_fast + t_val * w_full
    ce_a = (ce_char * w_fast).sum() / w_fast.sum().clamp(min=1e-8)
    ce_b = (ce_char * w_full).sum() / B4
    ce_mix = (1 - t_val) * ce_a + t_val * ce_b
    if t_val == 0.0:
        check(f"{name} ce=全幅 (实际{ce_mix:.4f})", abs(ce_mix - ce_a) < 1e-6)
    elif t_val == 1.0:
        check(f"{name} ce=固定分母 (实际{ce_mix:.4f})", abs(ce_mix - ce_b) < 1e-6)
    else:
        check(f"{name} ce∈[ce_b,ce_a] (实际{ce_mix:.4f})",
              min(ce_a, ce_b) - 1e-9 <= ce_mix <= max(ce_a, ce_b) + 1e-9)
    check(f"{name} w≥地板", (w >= W_FLOOR - 1e-9).all())
# 全幅性质: 权重全地板时 ce 仍是全幅均值 (死锁破除)
w_floor = _torch.full((1, 4), W_FLOOR)
ce_full = (ce_char * w_floor).sum() / w_floor.sum()
check(f"地板权重下全幅ce=均值 (实际{ce_full:.4f}, 期望{ce_char.mean():.4f})",
      abs(ce_full - ce_char.mean()) < 1e-6)

print("== 11. phase_c_t 端点 ==")
from config import phase_c_t
check("Ep299 → 未激活", phase_c_t(299) == (False, 0.0))
check("Ep300 → t=0 (硬CE)", phase_c_t(300) == (True, 0.0))
ph, t = phase_c_t(302)
check(f"Ep302 → t=0.4 (实际{t})", ph and abs(t - 0.4) < 1e-9)
ph, t = phase_c_t(305)
check(f"Ep305 → t=1 (实际{t})", ph and t == 1.0)
check("Ep400 → t=1", phase_c_t(400) == (True, 1.0))
check("禁用(0) → 未激活", phase_c_t(50, 0, 5) == (False, 0.0))
check("ramp=0 Ep300瞬切t=1", phase_c_t(300, 300, 0) == (True, 1.0))

print("== 12. 混淆软目标 CE ==")
import sys as _sys
_sys.path.insert(0, ".")
from train import build_soft_targets, soft_ce
import torch.nn.functional as _F
dist_mat, eps_vec = build_soft_targets()
check("eps∈[0,EPS_CAP]", bool((eps_vec >= 0).all() and (eps_vec <= EPS_CAP + 1e-9).all()))
check("无混淆数据字符 ε=0 (字符0)", float(eps_vec[CHARS.index("0")]) == 0.0)
check("I 有混淆分布", dist_mat[CHARS.index("I")].sum() > 0.5)
# t=0 ≡ 硬 CE
logits = _torch.randn(2, 4, NUM_CLASSES)
labels_t = _torch.randint(0, NUM_CLASSES, (2, 4))
hard = _F.cross_entropy(logits.reshape(-1, NUM_CLASSES), labels_t.reshape(-1),
                        reduction="none").view(2, 4)
soft0 = soft_ce(logits, labels_t, dist_mat, eps_vec, 0.0)
check(f"t=0≡硬CE (实际最大差{float((soft0-hard).abs().max()):.2e})",
      float((soft0 - hard).abs().max()) < 1e-5)
# t=1: 目标软化后 CE 改变 (ε>0 位置必有差异, Δ=log p_true-Σd·log p ≠ 0)
soft1 = soft_ce(logits, labels_t, dist_mat, eps_vec, 1.0)
mask = eps_vec[labels_t.reshape(-1)].view(2, 4) > 0
check(f"t=1 软CE≠硬CE (ε>0位置{int(mask.sum())}个, 差异{float((soft1-hard)[mask].abs().max()):.4f})",
      bool(mask.any() and (soft1 - hard)[mask].abs().max() > 1e-4))
# 数学性质: CE(t) 对 t 线性 → t=0.5 精确为 (硬+软)/2
soft5 = soft_ce(logits, labels_t, dist_mat, eps_vec, 0.5)
mid = 0.5 * (hard + soft1)
check(f"t=0.5 线性插值 (最大差{float((soft5-mid).abs().max()):.2e})",
      float((soft5 - mid).abs().max()) < 1e-5)
# ε=0 位置不随 t 变化 (字符0等无混淆数据字符)
if (~mask).any():
    check("ε=0位置恒=硬CE", float((soft1 - hard)[~mask].abs().max()) < 1e-6)
# 归一化: 目标分布每行和为1
_y = (1 - eps_vec[labels_t.reshape(-1)].unsqueeze(1)) * _F.one_hot(labels_t.reshape(-1), NUM_CLASSES).float() \
     + eps_vec[labels_t.reshape(-1)].unsqueeze(1) * dist_mat[labels_t.reshape(-1)]
check("软目标归一化", float((_y.sum(-1) - 1).abs().max()) < 1e-5)

print("== 13. gap 墨迹 hinge 损失 ==")
from train import gap_ink_loss, gap_ink_ratio
# 空白图: 无笔画 → loss=0
blank = _torch.ones(2, 1, 35, 90) * 0.95
check(f"空白图 gap=0 (实际{float(gap_ink_loss(blank)):.4f})",
      float(gap_ink_loss(blank)) < 0.05)
# 列界带画竖线: 超目标 → hinge>0 (ratio=1 → relu(1-0.5)=0.5)
lined = blank.clone()
for x0, x1 in GAP_STRIPS:
    lined[:, :, :, x0:x1] = -0.9
check(f"带内笔画 超目标受罚 (实际{float(gap_ink_loss(lined)):.4f})",
      float(gap_ink_loss(lined)) >= 0.45)
# 关键回归测试: 低于目标不再奖励 (无下界会把 gap 压到 0 并毁字)
half = blank.clone()
for x0, x1 in GAP_STRIPS:
    half[:, :, :, x0:x0 + 1] = -0.9    # 每带仅 1/4 列有墨 → ratio≈0.25 < 0.5
check(f"低于目标 hinge=0 不惩罚 (实际{float(gap_ink_loss(half)):.4f})",
      float(gap_ink_loss(half)) == 0.0)
# 带外笔画: gap 低 (字符列中心 x≈11,34,56,78)
centered = blank.clone()
for cx in (11, 34, 56, 78):
    centered[:, :, :, cx-3:cx+3] = -0.9
check(f"带外笔画 gap低 (实际{float(gap_ink_loss(centered)):.4f})",
      float(gap_ink_loss(centered)) < 0.15)
# 可导性
x = blank.clone().requires_grad_(True)
loss = gap_ink_loss(x)
loss.backward()
check("gap损失可导", x.grad is not None and bool(_torch.isfinite(x.grad).all()))
# ratio 硬判定口径
r = gap_ink_ratio(lined)
check(f"ratio>0.5 (实际{r:.3f})", r > 0.5)

print("== 14. Phase C 修正因子 ==")
# 非 Phase C: miss/miss 独苗照旧罚
w = apply_multipliers(0.20, "miss", "miss", "c_strong", 0.95, True, True, False)
check(f"常规模式 M_cheat 照旧 (实际{w:.3f})", abs(w - 0.10) < 1e-6)
# Phase C: miss/miss 但非垃圾 → 不罚 (目标态)
w = apply_multipliers(0.20, "miss", "miss", "c_strong", 0.95, True, True, False,
                      phase_c=True, d_garbage=False, p_garbage=False)
check(f"PhaseC 混淆miss不罚 (实际{w:.3f})", abs(w - 0.20) < 1e-6)
# Phase C: 双垃圾 → 罚
w = apply_multipliers(0.20, "miss", "miss", "c_strong", 0.95, True, True, False,
                      phase_c=True, d_garbage=True, p_garbage=True)
check(f"PhaseC 双垃圾罚 (实际{w:.3f})", abs(w - 0.10) < 1e-6)
# Phase C: 单垃圾 → 不罚 (保守)
w = apply_multipliers(0.20, "miss", "miss", "c_strong", 0.95, True, True, False,
                      phase_c=True, d_garbage=True, p_garbage=False)
check(f"PhaseC 单垃圾不罚 (实际{w:.3f})", abs(w - 0.20) < 1e-6)

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    raise SystemExit(1)
print("ALL TESTS PASSED")
