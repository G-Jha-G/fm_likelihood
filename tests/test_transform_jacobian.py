"""
tests.test_transform_jacobian
================================

Ports self-test 4 from fm_likelihood_extremes_v2.py's run_selftest: the
transform's analytic log-Jacobian, verified against a finite-difference
estimate. Unchanged from the pilot, this piece was already correct
(research programme Section 2: "Transform log-Jacobian ... correct").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fm_likelihood.transform import PrecipConfig, PrecipTransform


def test_log_jacobian_matches_finite_difference():
    cfg = PrecipConfig(log1p_max=6.7859)
    tr = PrecipTransform(cfg)
    torch.manual_seed(0)
    p = torch.rand(4, 1, 3, 3, dtype=torch.float64) * 50.0
    eps = 1e-6
    u1 = tr.to_model(p)
    u2 = tr.to_model(p + eps)
    fd = torch.log((u2 - u1).abs() / eps).flatten(1).sum(1)
    analytic = tr.log_dudp_sum(p)
    err = (fd - analytic).abs().max().item()
    assert err < 1e-4, f"max|finite_diff - analytic| = {err:.3e}"


def test_transform_is_invertible():
    """Not in the pilot's self-test, but cheap and directly relevant: if
    to_model/to_physical don't round-trip, the log-Jacobian is being
    computed for the wrong map."""
    cfg = PrecipConfig(log1p_max=6.7859)
    tr = PrecipTransform(cfg)
    torch.manual_seed(1)
    p = torch.rand(100, dtype=torch.float64) * 300.0
    u = tr.to_model(p)
    p_recovered = tr.to_physical(u)
    err = (p - p_recovered).abs().max().item()
    assert err < 1e-9, f"round trip error {err:.3e}"


def test_dequantize_stays_within_bin_and_nonnegative():
    cfg = PrecipConfig(log1p_max=6.7859, dequant_bin_mm=0.1, clip_min_mm=0.0)
    tr = PrecipTransform(cfg)
    p = torch.zeros(1000, dtype=torch.float64)  # worst case: all exact zeros
    gen = torch.Generator().manual_seed(0)
    p_deq = tr.dequantize(p, generator=gen)
    assert (p_deq >= 0).all()
    assert (p_deq < cfg.dequant_bin_mm).all()


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    tests = [f for name, f in inspect.getmembers(mod, inspect.isfunction)
             if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{len(tests)} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
