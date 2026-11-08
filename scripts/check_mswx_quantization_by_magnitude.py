#!/usr/bin/env python3
"""
check_mswx_quantization_by_magnitude.py
==========================================

Follow-up to check_mswx_quantization.py. That script found a minimum
nonzero spacing of 0.0625 mm globally, exactly float16's granularity at
64-128 mm/day, but a GLOBAL minimum can't distinguish two different
explanations: true float16 quantization (spacing scales with magnitude,
finer for small values, coarser for large ones) versus a uniform 1/16 mm
step applied everywhere regardless of magnitude. Those imply different
fixes: the former means no single dequant_bin_mm constant is correct at
all magnitudes; the latter means dequant_bin_mm should simply change
from 0.1 to 0.0625.

This bins wet values by power-of-2 magnitude range and computes the
minimum nonzero spacing WITHIN each bin separately. If spacing doubles
as magnitude doubles, that confirms float16-style scaling. If spacing is
constant (~0.0625) across all bins, that confirms uniform quantization.

USAGE
-----
    python scripts/check_mswx_quantization_by_magnitude.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_NC = "data/MSWX_Pcp_Daily_Ind_HR.nc"

FLOAT16_SPACING_AT_BIN_START = {}
for exp in range(-4, 9):
    lo = 2.0 ** exp
    FLOAT16_SPACING_AT_BIN_START[lo] = float(
        np.nextafter(np.float16(lo), np.float16(np.inf)) - np.float16(lo)
    )


def analyze(vals: np.ndarray, label: str) -> None:
    print(f"\n--- {label} ---")
    wet = np.sort(np.unique(vals[vals > 0]))
    print(f"{len(wet)} distinct positive values, range [{wet.min():.4f}, {wet.max():.4f}]")

    edges = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    print(f"{'bin':>16} {'n_values':>9} {'min spacing':>12} {'float16 predicts':>18}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = wet[(wet >= lo) & (wet < hi)]
        if len(in_bin) < 3:
            continue
        diffs = np.diff(in_bin)
        diffs = diffs[diffs > 1e-9]
        if len(diffs) == 0:
            continue
        min_sp = diffs.min()
        f16_pred = FLOAT16_SPACING_AT_BIN_START.get(lo, float("nan"))
        print(f"[{lo:>6g},{hi:>6g}) {len(in_bin):>9d} {min_sp:>12.6f} {f16_pred:>18.6f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nc", type=str, default=DEFAULT_NC)
    p.add_argument("--varname", type=str, default="precipitation")
    p.add_argument("--n-days-sample", type=int, default=60)
    args = p.parse_args()

    if not Path(args.nc).exists():
        print(f"error: {args.nc} not found. Pass --nc /path/to/file.nc")
        return 1

    print(f"Opening {args.nc}...")
    ds = xr.open_dataset(args.nc)
    da = ds[args.varname]
    n_total = da.shape[0]
    idx = np.linspace(0, n_total - 1, args.n_days_sample).astype(int)
    vals = da.isel({list(da.dims)[0]: idx}).values.astype(np.float64).ravel()
    vals = vals[np.isfinite(vals)]

    analyze(vals, "REAL DATA")

    print("\n" + "=" * 78)
    print("READING GUIDE")
    print("=" * 78)
    print("If 'min spacing' roughly DOUBLES each row (tracking the 'float16 predicts'")
    print("column closely): true magnitude-scaling quantization, likely a float16")
    print("rounding step somewhere upstream. No single dequant_bin_mm is correct at")
    print("all magnitudes; the dequantisation scheme itself would need to become")
    print("magnitude-aware, a real methodological change, not just a constant swap.")
    print()
    print("If 'min spacing' stays roughly CONSTANT (~0.0625) across all bins,")
    print("independent of the 'float16 predicts' column: uniform 1/16 mm quantisation.")
    print("Fix is then simple: dequant_bin_mm changes from 0.1 to 0.0625, everything")
    print("else in the pipeline is unaffected.")


if __name__ == "__main__":
    raise SystemExit(main())
