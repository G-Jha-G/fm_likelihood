"""
fm_likelihood.data
=====================

MSWX loading, with the India domain crop and the canonical period
boundaries in one place, specifically so neither has to be re-derived
(and re-risk the same bug) in every script that touches raw MSWX data.

THE DOMAIN CROP (verified 2026-08-11, do not re-derive by float labels)
--------------------------------------------------------------------------
Exact-label slicing (ds.sel(lat=slice(39.95, 5.05), ...)) silently drops
the southern edge row: MSWX_Pcp_Daily_Ind_HR.nc stores lat=5.05 as the
float32 value 5.0499983, which exact-match slicing on the float64 literal
5.05 does not catch, giving a (349, 350) crop instead of (350, 350). The
INDEX crop below was verified directly: cropping with these indices and
comparing day 0 (2011-01-01) against the pilot's mswx_test_2011-2014.h5
gave max abs diff = 0.0, i.e. byte-identical, not just matching shape.

INDIA_LAT_ISEL = slice(100, 450)   # -> lat 39.95 down to 5.05, 350 cells
INDIA_LON_ISEL = slice(150, 500)   # -> lon 65.05 up to 99.95, 350 cells

This also confirms, as a side effect, that mswx_test_2011-2014.h5's first
record really is 2011-01-01 with no day-offset.

CANONICAL PERIOD BOUNDARIES
------------------------------
FM and DDPM were trained and checkpointed against a specific chronological
split. Any new analysis that pulls MORE data from the full 1979 to 2025-06
record must respect this split explicitly, or "more test data" silently
becomes "some of it isn't actually held out":

    TRAIN_PERIOD      1979-01-01 to 2005-12-31   (model saw this)
    VALIDATION_PERIOD 2006-01-01 to 2010-12-31   (model selection saw this)
    TEST_PERIOD       2011-01-01 to 2014-12-31   (all evaluation in the
                                                    parent manuscript uses
                                                    this, 1461 daily fields)
    UNSEEN_PERIOD     2015-01-01 to 2025-06-03   (never touched by
                                                    training, validation,
                                                    or model selection at
                                                    all; a strictly
                                                    stronger holdout than
                                                    "test", but NOT the
                                                    same experiment as
                                                    TEST_PERIOD and should
                                                    not be silently pooled
                                                    with it)

The brief's EVT layer (Section 8.1) fits on 1980 to 2010, i.e. inside
TRAIN_PERIOD plus VALIDATION_PERIOD, which is correct for EVT (it needs
to be disjoint from the fields being SCORED, not disjoint from what the
downscaler trained on) but would be wrong to reuse as "test" data for
anything evaluating the model itself.
"""
from __future__ import annotations

from dataclasses import dataclass

INDIA_LAT_ISEL = slice(100, 450)
INDIA_LON_ISEL = slice(150, 500)
INDIA_DOMAIN_SHAPE = (350, 350)


@dataclass(frozen=True)
class PeriodBounds:
    start: str
    end: str


TRAIN_PERIOD = PeriodBounds("1979-01-01", "2005-12-31")
VALIDATION_PERIOD = PeriodBounds("2006-01-01", "2010-12-31")
TEST_PERIOD = PeriodBounds("2011-01-01", "2014-12-31")
UNSEEN_PERIOD = PeriodBounds("2015-01-01", "2025-06-03")
EVT_FIT_PERIOD = PeriodBounds("1980-01-01", "2010-12-31")  # brief Section 8.1


def crop_india_domain(ds, lat_dim: str = "lat", lon_dim: str = "lon"):
    """Apply the verified index crop to an xarray Dataset or DataArray.
    Uses isel (integer position), never sel (label match), specifically
    because label match on this file's float32 coordinates is what
    caused the original bug."""
    return ds.isel({lat_dim: INDIA_LAT_ISEL, lon_dim: INDIA_LON_ISEL})


def select_period(ds, period: PeriodBounds, time_dim: str = "time"):
    """Time-slice by an explicit PeriodBounds, so the period boundary is
    always named (TRAIN_PERIOD, TEST_PERIOD, ...) at the call site rather
    than a bare date string that's easy to get wrong or to silently let
    drift across TRAIN/VALIDATION/TEST/UNSEEN without noticing."""
    return ds.sel({time_dim: slice(period.start, period.end)})


def assert_no_period_overlap() -> None:
    """Sanity check on the boundaries themselves, run once at import time
    below: if these are ever edited, this catches an accidental overlap
    immediately rather than letting it silently leak train/val days into
    what a script thinks is disjoint test or EVT-fit data."""
    import datetime as _dt

    def _d(s: str) -> _dt.date:
        return _dt.date.fromisoformat(s)

    periods = [
        ("TRAIN", TRAIN_PERIOD), ("VALIDATION", VALIDATION_PERIOD),
        ("TEST", TEST_PERIOD), ("UNSEEN", UNSEEN_PERIOD),
    ]
    for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
            name_a, a = periods[i]
            name_b, b = periods[j]
            a_start, a_end = _d(a.start), _d(a.end)
            b_start, b_end = _d(b.start), _d(b.end)
            overlap = a_start <= b_end and b_start <= a_end
            assert not overlap, f"{name_a} and {name_b} overlap, check the boundaries"


assert_no_period_overlap()
