"""
tests.test_masks
==================

Tests fm_likelihood.mask at the project's real dimensions (H=W=350,
confirmed via Section 13 Q1), plus an empirical, not assumed, check that
the interior mask correctly separates duplicated pad cells from genuine
interior cells (finding N2).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from fm_likelihood.mask import compute_padding, interior_slices, interior_mask, to_interior


def test_padding_matches_pilot_arithmetic_at_real_dims():
    info = compute_padding(350, 350, div=16)
    assert info.pad_top == 1 and info.pad_bottom == 1
    assert info.pad_left == 1 and info.pad_right == 1
    assert info.h_padded == 352 and info.w_padded == 352


def test_d_interior_and_d_total_match_q1_answer():
    info = compute_padding(350, 350, div=16)
    assert info.d_interior == 122500
    assert info.d_total == 123904


def test_interior_mask_shape_and_count():
    info = compute_padding(350, 350, div=16)
    m = interior_mask(info)
    assert m.shape == (352, 352)
    assert int(m.sum()) == info.d_interior


def test_to_interior_recovers_original_field():
    """Pad a known field the same way the pilot does, then crop back
    with to_interior, and check we get the exact original back, not an
    off-by-one slice into the duplicated border."""
    H, W = 350, 350
    info = compute_padding(H, W, div=16)
    x = torch.arange(H * W, dtype=torch.float64).reshape(1, 1, H, W)
    padded = F.pad(x, (info.pad_left, info.pad_right, info.pad_top, info.pad_bottom),
                    mode='reflect')
    recovered = to_interior(padded, info)
    assert torch.equal(recovered, x)


def test_interior_mask_excludes_the_duplicated_border_cells():
    """Empirical, not assumed: build a field where every value is unique
    (arange), reflect-pad it, then confirm that every cell EXCLUDED by
    interior_mask is a duplicate of some cell INCLUDED by it, which is
    exactly what N2 says the pad region is. If this ever fails, the
    padding convention changed and mask.py needs to change with it."""
    H, W = 350, 350
    info = compute_padding(H, W, div=16)
    x = torch.arange(H * W, dtype=torch.float64).reshape(1, 1, H, W)
    padded = F.pad(x, (info.pad_left, info.pad_right, info.pad_top, info.pad_bottom),
                    mode='reflect')
    m = interior_mask(info)

    interior_values = set(padded[0, 0][m].tolist())
    pad_values = padded[0, 0][~m].tolist()
    for v in pad_values:
        assert v in interior_values, (
            f"pad cell with value {v} is not a duplicate of any interior value; "
            "the padding convention may have changed, mask.py assumptions are stale"
        )


def test_pad_fraction_is_small_but_nonzero():
    info = compute_padding(350, 350, div=16)
    assert 0 < info.pad_fraction < 0.02, \
        f"pad_fraction={info.pad_fraction:.4f}, sanity check it's still small"


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
