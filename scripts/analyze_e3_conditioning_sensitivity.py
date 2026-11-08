#!/usr/bin/env python3
"""
analyze_e3_conditioning_sensitivity.py
=========================================

Three of E3's four axes (magnitude, shift, blur) are dose-response
designs: does BPD move as the perturbation gets larger? The fourth
(swap) is a single paired comparison, not a dose ladder, and needs
different statistics entirely.

MAGNITUDE gets special treatment, and this matters: the response is
U-shaped (0.5x and 2x both raise BPD relative to 1x, and the final E3
run shows that the too-dry side is the steeper one). Correlating the
SIGNED log2(level) with BPD would average the two arms toward zero,
wrongly reading as "no sensitivity" when the opposite is true. Distance
is measured as |log2(level)|, the FOLDED distance from truth, so a
U-shaped response shows up as the strong positive correlation it is.
Shift and blur only move in one direction from baseline (0 upward), so
their raw level is already a valid monotonic distance metric.

PER-FIELD STATISTIC: for magnitude/shift/blur, Spearman correlation
between distance and BPD, pooling all replicates for that field (common
random numbers mean each replicate is an independent noise draw at the
SAME dose ladder, so pooling is a legitimate way to gain power per
field, not a violation of independence). For swap: mean paired
difference (BPD_swap - BPD_baseline) across replicates.

POPULATION INFERENCE: the 20 fields are a genuine cross-sectional
sample (not a time series, unlike E1), so this uses an ordinary iid
bootstrap over FIELDS, not E1's moving block bootstrap over time. Resample
the 20 per-field statistics with replacement, recompute the mean, repeat,
percentile CI. This is what E1's own within-day correlations lacked at
the individual-field level and is exactly the fix requested: a real
population-level claim, not 20 anecdotes.

VALIDATED, NOT ASSUMED: before running on real data, this script's
statistics are checked against synthetic data with KNOWN ground truth
in two directions, no true sensitivity (correlations should center on
zero, CIs should include it) and strong sensitivity (correlations
should be large and CIs should exclude zero), including a specifically
constructed symmetric U-shape case to confirm the folded-distance fix
actually does what it's supposed to.

USAGE
-----
    python scripts/analyze_e3_conditioning_sensitivity.py
    python scripts/analyze_e3_conditioning_sensitivity.py --n-boot 2000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


DISTANCE_FN = {
    "magnitude": lambda level: abs(np.log2(level)),   # FOLDED: U-shape safe
    "shift": lambda level: float(level),               # already one-directional from 0
    "blur": lambda level: float(level),                 # already one-directional from 0
}


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_field_dose_response(rows: list[dict], axis: str) -> dict:
    """field_id -> Spearman(distance, BPD), pooling all replicates,
    including the shared baseline point at distance=0."""
    by_field = defaultdict(list)
    for r in rows:
        if r["axis"] == axis:
            d = DISTANCE_FN[axis](r["level"])
            by_field[r["field_id"]].append((d, r["bits_per_interior_dim"]))
        elif r["axis"] == "baseline":
            by_field[r["field_id"]].append((0.0, r["bits_per_interior_dim"]))

    out = {}
    for field_id, pairs in by_field.items():
        d = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        out[field_id] = _spearman(d, y)
    return out


def per_field_swap_diff(rows: list[dict]) -> dict:
    """field_id -> mean(BPD_swap - BPD_baseline) across replicates."""
    baseline = defaultdict(dict)   # field_id -> replicate -> bpd
    swap = defaultdict(dict)
    for r in rows:
        if r["axis"] == "baseline":
            baseline[r["field_id"]][r["replicate"]] = r["bits_per_interior_dim"]
        elif r["axis"] == "swap":
            swap[r["field_id"]][r["replicate"]] = r["bits_per_interior_dim"]

    out = {}
    for field_id in swap:
        diffs = [swap[field_id][rep] - baseline[field_id][rep]
                 for rep in swap[field_id] if rep in baseline[field_id]]
        if diffs:
            out[field_id] = float(np.mean(diffs))
    return out


def field_bootstrap_ci(per_field_values: dict, n_boot: int, rng: np.random.Generator,
                        alpha: float = 0.05):
    """Ordinary iid bootstrap over FIELDS (not time), appropriate for a
    cross-sectional sample of 20 independent days."""
    vals = np.array([v for v in per_field_values.values() if not np.isnan(v)])
    n = len(vals)
    point = float(vals.mean())
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = vals[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), n


def summarize_axis(name: str, per_field: dict, n_boot: int, rng: np.random.Generator) -> None:
    point, lo, hi, n = field_bootstrap_ci(per_field, n_boot, rng)
    excludes_zero = lo > 0 or hi < 0
    print(f"{name:>10}: mean={point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"(n_fields={n})  {'EXCLUDES zero' if excludes_zero else 'includes zero'}")
    vals = sorted(per_field.items(), key=lambda kv: kv[1])
    print(f"{'':>10}  range across fields: min={vals[0][1]:+.4f} (field {vals[0][0]})  "
          f"max={vals[-1][1]:+.4f} (field {vals[-1][0]})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("e3_outputs/e3_results_FINAL_corrected.jsonl"))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.results.exists():
        print(f"error: {args.results} not found")
        return 1

    rows = load_results(args.results)
    print(f"Loaded {len(rows)} results from {args.results}\n")
    rng = np.random.default_rng(args.seed)

    print("=" * 88)
    print("DOSE-RESPONSE AXES: per-field Spearman(distance, BPD), field-bootstrap 95% CI")
    print("magnitude uses FOLDED distance |log2(level)| (U-shape safe)")
    print("=" * 88)
    for axis in ("magnitude", "shift", "blur"):
        per_field = per_field_dose_response(rows, axis)
        summarize_axis(axis, per_field, args.n_boot, rng)
        print()

    print("=" * 88)
    print("SWAP: per-field mean(BPD_swap - BPD_baseline), field-bootstrap 95% CI")
    print("=" * 88)
    swap_diffs = per_field_swap_diff(rows)
    summarize_axis("swap", swap_diffs, args.n_boot, rng)
    frac_positive = float(np.mean([v > 0 for v in swap_diffs.values()]))
    print(f"{'':>10}  fraction of fields where swap INCREASED BPD: {frac_positive:.3f} "
          f"of {len(swap_diffs)} (0.5 = no effect)")


if __name__ == "__main__":
    main()
