#!/usr/bin/env python3
"""
check_mswx_quantization.py
=============================

Q7, open since the very first session on this project: the CNF likelihood
pipeline assumes MSWX's real precision is 0.1 mm (PrecipConfig.dequant_bin_mm),
used for uniform dequantisation before every likelihood evaluation in this
study. That number was never verified, only assumed as a documented default.

This checks it two ways:
  1. File metadata: netCDF scale_factor/add_offset/dtype often state the
     product's true stored precision directly, no inference needed.
  2. Empirical: the minimum nonzero spacing between distinct values in a
     large sample of real precipitation values, which reveals quantisation
     directly regardless of what the metadata does or doesn't say.

USAGE
-----
    python scripts/check_mswx_quantization.py
    python scripts/check_mswx_quantization.py --nc /path/to/MSWX_Pcp_Daily_Ind_HR.nc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_NC = "data/MSWX_Pcp_Daily_Ind_HR.nc"


def check_metadata(ds: xr.Dataset, varname: str = "precipitation") -> None:
    print("=" * 78)
    print("1. FILE METADATA")
    print("=" * 78)
    da = ds[varname]
    print(f"dtype: {da.dtype}")
    for attr in ("scale_factor", "add_offset", "_FillValue", "missing_value",
                 "units", "precision", "resolution", "valid_min", "valid_max"):
        if attr in da.attrs:
            print(f"{attr}: {da.attrs[attr]}")
    if "scale_factor" in da.attrs:
        print(f"\n-> scale_factor present: stored precision is very likely "
              f"{da.attrs['scale_factor']} (the LSB of the packed integer representation)")
    else:
        print("\n-> no scale_factor attribute found; data may be stored as native "
              "float with no explicit packing, precision must come from the empirical check")

    print("\nDataset-level attributes:")
    for k, v in ds.attrs.items():
        print(f"  {k}: {v}")


def check_empirical_precision(ds: xr.Dataset, varname: str = "precipitation",
                               n_days_sample: int = 30) -> None:
    print("\n" + "=" * 78)
    print("2. EMPIRICAL CHECK (minimum nonzero spacing between distinct values)")
    print("=" * 78)
    da = ds[varname]
    n_total = da.sizes.get("time", None)
    if n_total is None:
        n_total = da.shape[0]
    idx = np.linspace(0, n_total - 1, n_days_sample).astype(int)
    print(f"Sampling {n_days_sample} days spread across the record "
          f"(indices {idx[0]}..{idx[-1]})...")

    vals = da.isel({list(da.dims)[0]: idx}).values.astype(np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    wet_vals = np.sort(np.unique(vals[vals > 0]))
    print(f"{len(vals)} total values sampled, {len(wet_vals)} distinct positive values")

    diffs = np.diff(wet_vals)
    diffs = diffs[diffs > 1e-9]
    min_diff = diffs.min()
    print(f"Minimum nonzero spacing between distinct sorted values: {min_diff:.6f} mm")

    print("\nFraction of sampled positive values consistent with being an exact "
          "multiple of each candidate precision (within floating-point tolerance):")
    for candidate in [0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]:
        remainder = np.abs(np.mod(wet_vals + candidate / 2, candidate) - candidate / 2)
        frac_consistent = float((remainder < candidate * 0.01).mean())
        print(f"  {candidate:>6.3f} mm: {frac_consistent*100:6.2f}% consistent")

    print(f"\nCurrent pipeline assumption (PrecipConfig.dequant_bin_mm): 0.1 mm")
    print("Compare the minimum spacing and the highest-scoring candidate above "
          "against this. If they agree, 0.1 mm is confirmed, not just assumed. "
          "If a finer precision (e.g. 0.01 mm) scores highest instead, the "
          "dequantisation bin should be tightened and E0f (dequant-bin sensitivity, "
          "never run) becomes worth doing before this number goes in the paper "
          "as settled.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nc", type=str, default=DEFAULT_NC)
    p.add_argument("--varname", type=str, default="precipitation")
    args = p.parse_args()

    if not Path(args.nc).exists():
        print(f"error: {args.nc} not found. Pass --nc /path/to/file.nc")
        return 1

    print(f"Opening {args.nc}...")
    ds = xr.open_dataset(args.nc)
    if args.varname not in ds:
        print(f"error: variable '{args.varname}' not in dataset. "
              f"Available: {list(ds.data_vars)}")
        return 1

    check_metadata(ds, args.varname)
    check_empirical_precision(ds, args.varname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
