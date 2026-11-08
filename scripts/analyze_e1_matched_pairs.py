#!/usr/bin/env python3
"""
analyze_e1_matched_pairs.py
==============================

E1 element 4, the last piece (brief Section E1): "Matched pair analysis:
pair fields with near identical wet fraction but different 99.9th
percentile intensity, and test the paired BPD difference. This is the
confound-free version of the pilot's Analysis B."

This is the direct replacement for the old, confounded "Analysis B"
(likelihood vs extremity) that Section 4.7 used to lean on: instead of
correlating BPD with extremity across all 1461 fields (where wet fraction
rides along and does most of the work), find pairs of fields with nearly
IDENTICAL wet fraction but very DIFFERENT extremity, and ask whether BPD
still differs between them. If it does, extremity has a real, confound-
free effect. If it doesn't, the raw correlation was wet fraction all
along.

METHOD
------
Greedy matching without replacement: sort fields by wet fraction, process
in descending order of extremity (so the fields with the most informative
contrast get first pick of available matches), pair each with the
unmatched field inside a wet-fraction caliper that maximizes the
extremity gap. No field appears in more than one pair.

Test statistic: Spearman correlation between the paired extremity
difference and the paired BPD difference (sign-consistent: always
higher-extremity member minus lower-extremity member within each pair).
A positive, non-zero correlation means larger extremity gaps predict
larger BPD gaps, at matched wet fraction.

UNCERTAINTY: the matching procedure itself, not just the final
correlation, is re-run inside a moving block bootstrap (same validated
block-length correction as e1_block_bootstrap.py), since the underlying
1461-day series is still temporally autocorrelated and a naive iid
resample of already-formed pairs would understate that.

USAGE
-----
    python scripts/analyze_e1_matched_pairs.py
    python scripts/analyze_e1_matched_pairs.py --caliper 0.02
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXTREMITY_COL = "p999_wet"
WF_COL = "wet_fraction"


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def sample_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    denom = float((x ** 2).sum())
    if denom == 0:
        return np.zeros(max_lag + 1)
    acf = np.empty(max_lag + 1)
    n = len(x)
    for k in range(max_lag + 1):
        acf[k] = float((x[: n - k] * x[k:]).sum()) / denom
    return acf


def decorrelation_time(x: np.ndarray, min_days: int = 15, max_lag: int = 200,
                        block_multiplier: float = 3.0) -> int:
    """Same validated 3x correction as e1_block_bootstrap.py, see that
    script's docstring for the validation details (85-92% coverage at 1x,
    ~90-95% at 3x, against known-truth synthetic AR(1) data)."""
    max_lag = min(max_lag, len(x) - 1)
    acf = sample_acf(x, max_lag)
    threshold = 1.0 / np.e
    below = np.where(np.abs(acf[1:]) < threshold)[0]
    tau = int(below[0]) + 1 if len(below) > 0 else max_lag
    return max(int(round(tau * block_multiplier)), min_days)


def moving_block_resample_indices(n: int, block_length: int, rng: np.random.Generator
                                   ) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block_length)) % n for s in starts])
    return idx[:n]


def match_pairs(df: pd.DataFrame, caliper: float, y_col: str) -> pd.DataFrame:
    """Greedy matching without replacement. Returns a dataframe of pairs
    with signed differences (high-extremity member minus low-extremity
    member for both extremity and BPD)."""
    work = df[["field_id", WF_COL, EXTREMITY_COL, y_col]].copy()
    work = work.sort_values(EXTREMITY_COL, ascending=False).reset_index(drop=True)
    wf = work[WF_COL].to_numpy()
    ext = work[EXTREMITY_COL].to_numpy()
    y = work[y_col].to_numpy()
    n = len(work)
    matched = np.zeros(n, dtype=bool)

    pairs = []
    for i in range(n):
        if matched[i]:
            continue
        candidates = np.where((~matched) & (np.abs(wf - wf[i]) <= caliper))[0]
        candidates = candidates[candidates != i]
        if len(candidates) == 0:
            continue
        gaps = np.abs(ext[candidates] - ext[i])
        best = candidates[np.argmax(gaps)]
        if ext[i] >= ext[best]:
            hi, lo = i, best
        else:
            hi, lo = best, i
        pairs.append({
            "field_id_hi": int(work.loc[hi, "field_id"]),
            "field_id_lo": int(work.loc[lo, "field_id"]),
            "wf_hi": wf[hi], "wf_lo": wf[lo],
            "ext_hi": ext[hi], "ext_lo": ext[lo],
            "delta_ext": ext[hi] - ext[lo],
            "delta_bpd": y[hi] - y[lo],
        })
        matched[i] = True
        matched[best] = True

    return pd.DataFrame(pairs)


def matched_pair_statistic(df: pd.DataFrame, caliper: float, y_col: str) -> float:
    pairs = match_pairs(df, caliper, y_col)
    if len(pairs) < 5:
        return float("nan")
    return _spearman(pairs["delta_ext"].to_numpy(), pairs["delta_bpd"].to_numpy())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=Path, default=Path("e1_analysis_table.csv"))
    p.add_argument("--y-col", type=str, default="bits_per_interior_dim")
    p.add_argument("--caliper", type=float, default=0.01,
                   help="max allowed wet_fraction difference within a matched pair")
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.table.exists():
        print(f"error: {args.table} not found")
        return 1

    df = pd.read_csv(args.table).sort_values("field_id").reset_index(drop=True)
    print(f"Loaded {len(df)} rows, caliper={args.caliper} on wet_fraction\n")

    pairs = match_pairs(df, args.caliper, args.y_col)
    n_pairs = len(pairs)
    n_unmatched = len(df) - 2 * n_pairs
    print(f"Formed {n_pairs} pairs ({2*n_pairs} of {len(df)} fields matched, "
          f"{n_unmatched} left unmatched, no caliper match available)")
    print(f"Mean wet_fraction gap within pairs: {(pairs['wf_hi']-pairs['wf_lo']).abs().mean():.4f} "
          f"(caliper was {args.caliper})")
    print(f"Mean extremity gap within pairs: {pairs['delta_ext'].mean():.2f} mm")
    print(f"Fraction of pairs where higher-extremity member also has higher BPD: "
          f"{(pairs['delta_bpd'] > 0).mean():.3f} (0.5 = no relationship, 1.0 = always higher)\n")

    point_rho = _spearman(pairs["delta_ext"].to_numpy(), pairs["delta_bpd"].to_numpy())
    print(f"Point estimate, Spearman(delta_extremity, delta_BPD) across {n_pairs} pairs: "
          f"{point_rho:.4f}\n")

    print("Running block bootstrap over the matching + testing procedure "
          "(re-matches on each resample, not just resampling pre-formed pairs)...")
    y = df[args.y_col].to_numpy(dtype=np.float64)
    tau = decorrelation_time(y, min_days=15)
    print(f"Block length: {tau} days, n_boot={args.n_boot}")

    rng = np.random.default_rng(args.seed)
    n = len(df)
    boot_vals = []
    for b in range(args.n_boot):
        idx = moving_block_resample_indices(n, tau, rng)
        df_b = df.iloc[idx].reset_index(drop=True)
        stat = matched_pair_statistic(df_b, args.caliper, args.y_col)
        if not np.isnan(stat):
            boot_vals.append(stat)
    boot_vals = np.array(boot_vals)
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])

    print(f"\n{'=' * 88}")
    print("MATCHED-PAIR RESULT (confound-free version of the pilot's Analysis B)")
    print("=" * 88)
    print(f"Spearman(delta_extremity, delta_BPD) = {point_rho:.4f}   "
          f"95% CI [{lo:.4f}, {hi:.4f}]   (n_boot successful: {len(boot_vals)}/{args.n_boot})")
    excludes_zero = lo > 0 or hi < 0
    print(f"CI {'EXCLUDES' if excludes_zero else 'INCLUDES'} zero.")
    print("\nSame caveat as e1_block_bootstrap.py: this CI's coverage was validated at "
          "~90-95%, not exactly nominal, treat 'includes zero' as moderate rather than "
          "strong evidence either way.")
    if excludes_zero and lo > 0:
        print("\nReading: at matched wet fraction, larger extremity gaps predict larger BPD "
              "gaps. This is the direct, confound-free complement to the partial correlation "
              "in analyze_e1_confound_decomposition.py, same conclusion, different method, "
              "no wet-fraction adjustment via regression needed here since it's controlled "
              "by construction (the matching itself).")


if __name__ == "__main__":
    main()
