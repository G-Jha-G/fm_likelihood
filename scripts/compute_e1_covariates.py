#!/usr/bin/env python3
"""
compute_e1_covariates.py
===========================

E1's covariate table (brief Section E1): dry fraction, wet fraction, domain
mean, wet day mean, 99.9th percentile of wet pixels, spatial max, RAPS log
spectral slope, connected wet component count and mean size, intensity
concentration, Moran's I, season, day of year, for every day in the test
period.

CPU-only, no torch, no GPU, no trained model. Safe to run in parallel with
any GPU-bound script, including one already using the card. Reads directly
from mswx_test_2011-2014.h5, which already contains all 1461 days (the
pilot's `[::10]` was a loading-time subsample choice, not a limit on what's
stored, confirmed via the earlier grid-dimensions check).

ASSUMPTIONS WORTH KNOWING ABOUT
----------------------------------
1. Dates: constructed as a plain daily range from 2011-01-01 for 1461
   periods (365+366+365+365, correctly spans the 2012 leap year), not read
   from a stored coordinate, the h5 file only has a bare 'precipitation'
   array. This matches record 0 = 2011-01-01, independently confirmed
   earlier by comparing against the raw MSWX netCDF. Worth a quick spot
   check on a couple of interior dates if you want extra confidence, but
   there is no known reason to expect a gap or duplicate in this file.

2. RAPS spectral slope is computed on log1p(precip), not raw mm. Raw
   precipitation's huge dynamic range and dominant zero mass make a
   spectrum computed directly on it dominated by the marginal, not spatial
   structure. log1p matches the model's own preprocessing transform.
   Fitted over wavenumbers 2 to N/4 (excludes the noisy DC-adjacent bin
   and the aliasing-prone region near Nyquist), on the INTERIOR domain
   only (350x350, unpadded, since padding is duplicated data that would
   distort a spectral estimate).

3. Moran's I uses simple rook (4-neighbor) contiguity with unit weights,
   computed via array shifts rather than an explicit weight matrix (which
   would be (350*350)^2, intractable), standard practice for regular
   raster grids.

4. Wet threshold is 1.0 mm, matching PrecipConfig.wet_threshold_mm and
   the definition used throughout E0's field selection.

USAGE
-----
    python scripts/compute_e1_covariates.py
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

WET_THRESHOLD_MM = 1.0
MSWX_TEST_H5 = os.environ.get("MSWX_TEST_H5", "data/mswx_test_2011-2014.h5")
def season_of(date: pd.Timestamp) -> str:
    m = date.month
    if m in (6, 7, 8, 9):
        return "JJAS"
    if m in (3, 4, 5):
        return "MAM"
    if m in (10, 11, 12):
        return "OND"
    return "JF"  # 1, 2


def raps_log_slope(field_mm: np.ndarray) -> float:
    """Radially averaged power spectrum, log-log slope, on log1p(field).
    Negative for typical geophysical fields; more negative means a
    steeper falloff, i.e. a smoother field with less small-scale
    structure."""
    x = np.log1p(field_mm)
    x = x - x.mean()
    H, W = x.shape
    F = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(F) ** 2

    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_mean = np.zeros(max_r)
    for k in range(max_r):
        mask = r == k
        if mask.any():
            radial_mean[k] = power[mask].mean()

    lo, hi = 2, max(3, max_r // 4)
    k_band = np.arange(lo, hi)
    p_band = radial_mean[lo:hi]
    valid = p_band > 0
    if valid.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(k_band[valid]), np.log(p_band[valid]), 1)
    return float(slope)


def morans_i(field: np.ndarray) -> float:
    """Rook contiguity, unit weights, computed via array shifts (a full
    (H*W)^2 weight matrix is intractable at this grid size)."""
    x = field.astype(np.float64)
    dx = x - x.mean()
    h_pairs = (dx[:, :-1] * dx[:, 1:]).sum()
    v_pairs = (dx[:-1, :] * dx[1:, :]).sum()
    cross = 2.0 * (h_pairs + v_pairs)
    n_h = dx[:, :-1].size
    n_v = dx[:-1, :].size
    W_sum = 2.0 * (n_h + n_v)
    n = dx.size
    denom = (dx ** 2).sum()
    if denom == 0 or W_sum == 0:
        return float("nan")
    return float((n / W_sum) * (cross / denom))


def connected_wet_components(field_mm: np.ndarray, threshold: float = WET_THRESHOLD_MM):
    wet_mask = field_mm > threshold
    labeled, n_components = ndimage.label(wet_mask, structure=np.ones((3, 3)))
    if n_components == 0:
        return 0, 0.0
    sizes = ndimage.sum(wet_mask, labeled, index=range(1, n_components + 1))
    return int(n_components), float(np.mean(sizes))


def intensity_concentration_top1pct(field_mm: np.ndarray) -> float:
    flat = field_mm.flatten()
    total = flat.sum()
    if total <= 0:
        return float("nan")
    n_top = max(1, int(0.01 * flat.size))
    top_sum = np.partition(flat, -n_top)[-n_top:].sum()
    return float(top_sum / total)


def compute_covariates_for_field(field_mm: np.ndarray, date: pd.Timestamp) -> dict:
    wet_mask = field_mm > WET_THRESHOLD_MM
    dry_mask = field_mm == 0.0
    wet_vals = field_mm[wet_mask]

    n_components, mean_component_size = connected_wet_components(field_mm)

    return {
        "date": date.strftime("%Y-%m-%d"),
        "day_of_year": int(date.dayofyear),
        "season": season_of(date),
        "dry_fraction": float(dry_mask.mean()),
        "wet_fraction": float(wet_mask.mean()),
        "domain_mean": float(field_mm.mean()),
        "wet_day_mean": float(wet_vals.mean()) if wet_vals.size > 0 else 0.0,
        "p999_wet": float(np.percentile(wet_vals, 99.9)) if wet_vals.size > 0 else 0.0,
        "spatial_max": float(field_mm.max()),
        "raps_log_slope": raps_log_slope(field_mm),
        "n_connected_wet_components": n_components,
        "mean_component_size": mean_component_size,
        "intensity_concentration_top1pct": intensity_concentration_top1pct(field_mm),
        "morans_i": morans_i(field_mm),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=str, default=MSWX_TEST_H5)
    p.add_argument("--out", type=Path, default=Path("e1_covariates.csv"))
    p.add_argument("--n-fields", type=int, default=None, help="limit for a quick test run")
    args = p.parse_args()

    print(f"Loading {args.h5}...")
    with h5py.File(args.h5, "r") as f:
        precip = f["precipitation"][:]
    n_total = precip.shape[0]
    n = args.n_fields or n_total
    print(f"{n_total} days in file, computing covariates for the first {n}")

    dates = pd.date_range("2011-01-01", periods=n_total, freq="D")
    if n_total != 1461:
        print(f"[warn] expected 1461 days, got {n_total}. Season/day-of-year assignment "
              f"still assumes a plain daily sequence from 2011-01-01, verify this is right "
              f"for your file before trusting the season column.")

    rows = []
    for i in range(n):
        field = precip[i]
        row = {"field_id": i, **compute_covariates_for_field(field, dates[i])}
        rows.append(row)
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"[{i + 1}/{n}]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(df)} rows, {len(df.columns)} columns)")
    print("\nPer-season counts:")
    print(df["season"].value_counts())
    print("\nSummary of a few key covariates:")
    print(df[["dry_fraction", "wet_fraction", "spatial_max", "raps_log_slope",
              "morans_i"]].describe())


if __name__ == "__main__":
    main()
