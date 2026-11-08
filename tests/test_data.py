"""
tests.test_data
=================

Tests fm_likelihood.data's crop and period logic against a synthetic
dataset shaped like the real MSWX file (500x500, float32 coords with a
boundary value that doesn't land on a clean float, same as the real bug),
so the crop is exercised the same way it failed the first time, not just
on convenient round numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr

from fm_likelihood.data import (
    INDIA_LAT_ISEL, INDIA_LON_ISEL, INDIA_DOMAIN_SHAPE,
    TRAIN_PERIOD, VALIDATION_PERIOD, TEST_PERIOD, UNSEEN_PERIOD, EVT_FIT_PERIOD,
    crop_india_domain, select_period, assert_no_period_overlap,
)


def _make_synthetic_mswx_like():
    """500x500 grid, float32 coords, same 0.1 degree spacing as the real
    file, deliberately including the exact float32 rounding that broke
    exact-label slicing the first time (lat=5.05 stored as 5.0499983)."""
    lat = np.round(np.arange(49.95, -0.05, -0.1), 2).astype(np.float32)
    lon = np.round(np.arange(50.05, 100.05, 0.1), 2).astype(np.float32)
    time = xr.cftime_range("1979-01-01", periods=30, freq="D") \
        if False else np.array([f"1979-01-{d:02d}" for d in range(1, 31)],
                                dtype="datetime64[ns]")
    data = np.arange(len(time) * len(lat) * len(lon), dtype=np.float32).reshape(
        len(time), len(lat), len(lon))
    return xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_crop_gives_correct_shape():
    ds = _make_synthetic_mswx_like()
    cropped = crop_india_domain(ds)
    assert cropped.precipitation.shape[-2:] == INDIA_DOMAIN_SHAPE


def test_crop_uses_isel_not_sel_boundary_survives():
    """The actual bug this module exists to prevent: exact-label .sel()
    silently drops the southern boundary row when the stored float32
    value isn't bit-identical to the requested float64 literal. isel()
    doesn't have this problem since it's a pure integer position."""
    ds = _make_synthetic_mswx_like()
    lat_vals = ds.lat.values
    exact_matches = np.sum(lat_vals == np.float32(5.05))
    cropped = crop_india_domain(ds)
    assert cropped.precipitation.shape[-2:] == (350, 350)


def test_crop_indices_match_documented_values():
    assert INDIA_LAT_ISEL == slice(100, 450)
    assert INDIA_LON_ISEL == slice(150, 500)


def test_select_period_returns_correct_range():
    ds = _make_synthetic_mswx_like()
    subset = select_period(ds, TRAIN_PERIOD)
    times = subset.time.values
    assert len(times) == 30
    assert str(times.min())[:10] == "1979-01-01"


def test_no_overlap_among_train_val_test_unseen():
    assert_no_period_overlap()


def test_evt_fit_period_deliberately_overlaps_train_and_validation():
    """Intentional, documented behaviour, not a bug: EVT fitting needs to
    be disjoint from the fields being SCORED (the test period), not
    disjoint from what the model trained on."""
    import datetime as dt
    evt_start = dt.date.fromisoformat(EVT_FIT_PERIOD.start)
    evt_end = dt.date.fromisoformat(EVT_FIT_PERIOD.end)
    train_end = dt.date.fromisoformat(TRAIN_PERIOD.end)
    val_start = dt.date.fromisoformat(VALIDATION_PERIOD.start)
    assert evt_start <= train_end
    assert evt_end >= val_start

    test_start = dt.date.fromisoformat(TEST_PERIOD.start)
    assert evt_end < test_start


def test_unseen_period_starts_after_test_ends():
    import datetime as dt
    test_end = dt.date.fromisoformat(TEST_PERIOD.end)
    unseen_start = dt.date.fromisoformat(UNSEEN_PERIOD.start)
    assert unseen_start > test_end


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
