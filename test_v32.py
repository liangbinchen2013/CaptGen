# -*- coding: utf-8 -*-
"""v32 单元测试 (CPU, 不加载第三方 OCR).

用法: python test_v32.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def test_len_fail_weight():
    from adaptive_ocr import (apply_multipliers, C_STRONG, DP_MISS, DP_HIT)
    from config import LEN_FAIL_MIN_WEIGHT
    # 旧行为: C强 + 双miss + 双长度错 -> 0.20*0.85^2*0.5 -> 0.08 地板
    w_old = apply_multipliers(0.20, DP_MISS, DP_MISS, C_STRONG, 0.99,
                              False, False, False, len_fail=False)
    w_new = apply_multipliers(0.20, DP_MISS, DP_MISS, C_STRONG, 0.99,
                              False, False, False, len_fail=True)
    w_hit = apply_multipliers(1.00, DP_HIT, DP_HIT, C_STRONG, 0.99,
                              True, True, False)
    print(f"  old_len_fail={w_old:.4f} new_len_fail={w_new:.4f} triple_hit={w_hit:.4f}")
    assert abs(w_old - 0.08) < 1e-9
    assert abs(w_new - LEN_FAIL_MIN_WEIGHT) < 1e-9
    assert abs(w_hit - 1.0) < 1e-9
    print("  PASS len_fail skip M_LEN/M_CHEAT + min weight")


def test_fast_path_mask():
    from adaptive_ocr import MultiOCRAdaptive
    c = MultiOCRAdaptive(device="cpu", load_third_party=False)
    imgs = torch.randn(4, 1, 35, 90)
    labels = torch.randint(1, 36, (4, 4))
    _, info = c.compute_char_weights(imgs, labels, max_samples=0)
    print(f"  fast len_fail_mask sum={int(info['len_fail_mask'].sum())}")
    assert info["len_fail_mask"].shape == (4,)
    assert int(info["len_fail_mask"].sum()) == 0
    print("  PASS fast path mask all False")


def test_spatial_loss():
    import ocr_net
    a = torch.randn(4, 8, 4, 6)
    b = torch.randn(4, 8, 4, 6)
    l0 = float(ocr_net.ocr_spatial_loss(a, a))
    l1 = float(ocr_net.ocr_spatial_loss(a, b))
    print(f"  same={l0:.6f} diff={l1:.6f}")
    assert l0 < 1e-6 and l1 > 0
    print("  PASS spatial loss")


def test_real_labels():
    import train
    x = train.sample_real_labels(256, "cpu")
    print(f"  zeros={int((x == 0).sum())} min={int(x.min())}")
    assert (x == 0).sum() == 0
    assert x.shape == (256, 4)
    print("  PASS real label sampler excludes '0'")


def test_column_profile():
    import torch.nn.functional as F
    from train import column_ink_profile
    a = torch.rand(3, 1, 35, 90) * 2 - 1
    b = torch.rand(3, 1, 35, 90) * 2 - 1
    p0 = column_ink_profile(a)
    p1 = column_ink_profile(b)
    l0 = float(F.l1_loss(p0, p0))
    l1 = float(F.l1_loss(p0, p1))
    print(f"  same={l0:.6f} diff={l1:.6f}")
    assert p0.shape == (90,)
    assert l0 < 1e-6 and l1 > 0
    print("  PASS column ink profile")


def test_pair_key():
    from evaluate_generated import split_pair_key
    assert split_pair_key("<LEN>>4->3") == ("<LEN>", "4->3")
    assert split_pair_key("7>>") == ("7", ">")
    assert split_pair_key("O>0") == ("O", "0")
    print("  PASS split_pair_key")


if __name__ == "__main__":
    test_len_fail_weight()
    test_fast_path_mask()
    test_spatial_loss()
    test_column_profile()
    test_real_labels()
    test_pair_key()
    print("ALL PASS")
