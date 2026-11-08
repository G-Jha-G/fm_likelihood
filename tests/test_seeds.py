"""
tests.test_seeds
=================

These tests exist because the entire statistical protocol (Section 8 of
the brief: paired designs, common random numbers, block bootstrap) is
only valid if rng_key is deterministic, injective in practice, and does
not collide across the axes that are supposed to distinguish draws.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm_likelihood.seeds import rng_key, torch_generator, numpy_rng


def test_deterministic_same_inputs_same_key():
    a = rng_key(field_id=17, condition="matched", replicate=0, purpose="hutchinson")
    b = rng_key(field_id=17, condition="matched", replicate=0, purpose="hutchinson")
    assert a == b


def test_field_id_changes_key():
    a = rng_key(field_id=1, condition="matched", replicate=0, purpose="hutchinson")
    b = rng_key(field_id=2, condition="matched", replicate=0, purpose="hutchinson")
    assert a != b


def test_condition_changes_key():
    a = rng_key(field_id=1, condition="matched", replicate=0, purpose="hutchinson")
    b = rng_key(field_id=1, condition="mismatched", replicate=0, purpose="hutchinson")
    assert a != b


def test_replicate_changes_key():
    a = rng_key(field_id=1, condition="matched", replicate=0, purpose="hutchinson")
    b = rng_key(field_id=1, condition="matched", replicate=1, purpose="hutchinson")
    assert a != b


def test_purpose_changes_key():
    a = rng_key(field_id=1, condition="matched", replicate=0, purpose="hutchinson")
    b = rng_key(field_id=1, condition="matched", replicate=0, purpose="dequant")
    assert a != b


def test_common_random_numbers_pattern():
    """The E2/E3 pairing trick: same field_id, same replicate, same purpose,
    same condition string shared between the two members of a pair, must
    give the identical key so the paired draw truly is shared."""
    key_for_A = rng_key(field_id=42, condition="shared", replicate=0, purpose="dequant")
    key_for_B = rng_key(field_id=42, condition="shared", replicate=0, purpose="dequant")
    assert key_for_A == key_for_B


def test_no_collisions_across_147_fields_x_4_purposes():
    """Not a proof of injectivity, but a concrete check at the scale this
    project actually uses (147 test fields, 4 known purposes): if any two
    of these 588 keys collided, two supposedly independent draws would be
    identical, silently breaking the numerical noise floor estimate in
    E0a. Zero collisions expected; blake2b at digest_size=8 mod 2**31-1
    over this small a sample should never collide in practice."""
    keys = set()
    n_fields = 147
    for field_id in range(n_fields):
        for purpose in ("dequant", "hutchinson", "surrogate", "mismatch"):
            k = rng_key(field_id=field_id, condition="matched", replicate=0, purpose=purpose)
            assert k not in keys, f"collision at field_id={field_id}, purpose={purpose}"
            keys.add(k)
    assert len(keys) == n_fields * 4


def test_key_in_valid_range():
    for field_id in (0, 1, 146, 999999):
        k = rng_key(field_id=field_id, condition="x", replicate=0, purpose="dequant")
        assert 0 <= k < 2**31 - 1


def test_torch_generator_reproducible():
    import torch
    g1 = torch_generator(field_id=5, condition="matched", replicate=0, purpose="hutchinson")
    g2 = torch_generator(field_id=5, condition="matched", replicate=0, purpose="hutchinson")
    t1 = torch.randint(0, 2, (100,), generator=g1)
    t2 = torch.randint(0, 2, (100,), generator=g2)
    assert torch.equal(t1, t2)


def test_numpy_rng_reproducible():
    r1 = numpy_rng(field_id=5, condition="matched", replicate=0, purpose="surrogate")
    r2 = numpy_rng(field_id=5, condition="matched", replicate=0, purpose="surrogate")
    assert (r1.permutation(1000) == r2.permutation(1000)).all()


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    tests = [f for name, f in inspect.getmembers(mod, inspect.isfunction)
             if name.startswith("test_")]
    passed, skipped = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except ModuleNotFoundError as e:
            print(f"SKIP  {t.__name__}: {e} (expected on a machine without this dependency)")
            skipped += 1
    print(f"\n{passed}/{len(tests)} passed, {skipped} skipped, "
          f"{len(tests) - passed - skipped} failed")
    sys.exit(0 if passed + skipped == len(tests) else 1)
