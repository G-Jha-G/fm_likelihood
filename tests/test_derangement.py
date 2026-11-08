"""
tests.test_derangement
========================

Directly tests the N9 fix: derangement() must never return a permutation
with a fixed point, checked exhaustively at the project's actual scale
(n=147) over many repeated draws, not just a handful.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fm_likelihood.derangement import derangement
from fm_likelihood.seeds import torch_generator


def test_is_a_valid_permutation():
    gen = torch.Generator().manual_seed(0)
    perm = derangement(147, generator=gen)
    assert sorted(perm.tolist()) == list(range(147))


def test_no_fixed_points_single_draw():
    gen = torch.Generator().manual_seed(0)
    perm = derangement(147, generator=gen)
    assert not (perm == torch.arange(147)).any()


def test_no_fixed_points_over_many_seeds():
    """The actual N9 stress test: 500 independent draws at n=147, zero
    fixed points every time, by construction rather than by luck."""
    for seed in range(500):
        gen = torch.Generator().manual_seed(seed)
        perm = derangement(147, generator=gen)
        assert not (perm == torch.arange(147)).any(), f"fixed point at seed={seed}"


def test_deterministic_given_same_generator_key():
    gen1 = torch_generator(field_id=-1, condition="derangement", replicate=0,
                            purpose="mismatch")
    gen2 = torch_generator(field_id=-1, condition="derangement", replicate=0,
                            purpose="mismatch")
    p1 = derangement(147, generator=gen1)
    p2 = derangement(147, generator=gen2)
    assert torch.equal(p1, p2)


def test_raises_on_n_less_than_2():
    try:
        derangement(1, generator=torch.Generator().manual_seed(0))
        assert False, "expected ValueError for n=1"
    except ValueError:
        pass


def test_n_equals_2_works():
    """The only derangement of {0,1} is the swap [1,0], must always
    return it."""
    gen = torch.Generator().manual_seed(0)
    perm = derangement(2, generator=gen)
    assert perm.tolist() == [1, 0]


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
