#!/usr/bin/env python3
"""
analyze_e3_blur_corrected.py
===============================

Corrects a real mistake: analyze_e3_conditioning_sensitivity.py's single
Spearman correlation for blur (rho=0.019) understated the true effect,
because the actual response isn't monotonic. Looking at the real data
directly (not just the summary statistic) shows a remarkably clean,
two-phase pattern: mild blur (sigma 1-4) DECREASES BPD in 20 of 20
fields (mean delta -0.011 to -0.043), and aggressive blur (sigma 8)
INCREASES BPD in 20 of 20 fields (mean delta +0.069), the opposite
direction. A monotonic correlation averages these two consistent,
opposite-signed effects toward zero. Neither phase is noise; each is
essentially unanimous across the field sample. The values quoted here are for
the corrected reflect-padded blur (e3_results_FINAL_corrected.jsonl); the
superseded zero-padded blur gave +0.081 at sigma 8.

This replaces the single correlation with what the shape actually
calls for: a descriptive table across all 5 blur levels (pooled mean
delta with a field-bootstrap CI at each level, not just SEM), plus two
explicit paired comparisons, mirroring the swap axis's own paired
design, since a dip-then-rise shape is closer to "two effects" than to
"one dose-response":

  MILD BLUR  (sigma=4, near the visible minimum): does gentle smoothing
             of the conditioning make the field look MORE plausible?
  AGGRESSIVE BLUR (sigma=8): does heavy smoothing destroy enough real
             structure to make the field look LESS plausible?

Uses the same field-level iid bootstrap as the main E3 analysis script
(already coverage-validated there), applied to these two new paired
statistics.

USAGE
-----
    python scripts/analyze_e3_blur_corrected.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_field_baseline(rows: list[dict]) -> dict:
    by_field = defaultdict(list)
    for r in rows:
        if r["axis"] == "baseline":
            by_field[r["field_id"]].append(r["bits_per_interior_dim"])
    return {f: float(np.mean(v)) for f, v in by_field.items()}


def per_field_level_diff(rows: list[dict], axis: str, level, baseline: dict) -> dict:
    """field_id -> mean(BPD_at_level - BPD_baseline) across replicates."""
    by_field = defaultdict(list)
    for r in rows:
        if r["axis"] == axis and r["level"] == level:
            by_field[r["field_id"]].append(r["bits_per_interior_dim"])
    return {f: float(np.mean(v)) - baseline[f] for f, v in by_field.items() if f in baseline}


def field_bootstrap_ci(per_field_values: dict, n_boot: int, rng: np.random.Generator,
                        alpha: float = 0.05):
    vals = np.array([v for v in per_field_values.values() if not np.isnan(v)])
    n = len(vals)
    point = float(vals.mean())
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = vals[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("e3_outputs/e3_results_FINAL_corrected.jsonl"))
    p.add_argument("--n-boot", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.results.exists():
        print(f"error: {args.results} not found")
        return 1

    rows = load_results(args.results)
    print(f"Loaded {len(rows)} results\n")
    baseline = per_field_baseline(rows)
    rng = np.random.default_rng(args.seed)

    print("=" * 88)
    print("DESCRIPTIVE: pooled mean(BPD - own baseline) at each blur level, field-bootstrap CI")
    print("=" * 88)
    print(f"{'sigma':>6} {'n':>3} {'mean_delta':>11} {'95% CI':>22} {'n_negative':>11}")
    for level in (1, 2, 4, 8):
        per_field = per_field_level_diff(rows, "blur", level, baseline)
        point, lo, hi, n = field_bootstrap_ci(per_field, args.n_boot, rng)
        n_neg = sum(1 for v in per_field.values() if v < 0)
        print(f"{level:>6} {n:>3} {point:>+11.4f}   [{lo:+.4f}, {hi:+.4f}]   "
              f"{n_neg:>4}/{n} negative")

    print()
    print("=" * 88)
    print("PAIRED COMPARISON 1: MILD BLUR (sigma=4) vs baseline")
    print("Does gentle smoothing make the field look MORE plausible (negative delta)?")
    print("=" * 88)
    mild = per_field_level_diff(rows, "blur", 4, baseline)
    point, lo, hi, n = field_bootstrap_ci(mild, args.n_boot, rng)
    excludes_zero = lo > 0 or hi < 0
    print(f"mean delta = {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   n_fields={n}")
    print(f"{'EXCLUDES zero' if excludes_zero else 'includes zero'}"
          + (", direction: DECREASE (more plausible)" if point < 0 else ""))
    print(f"Fraction of fields with negative delta: "
          f"{sum(1 for v in mild.values() if v < 0)}/{n}")

    print()
    print("=" * 88)
    print("PAIRED COMPARISON 2: AGGRESSIVE BLUR (sigma=8) vs baseline")
    print("Does heavy smoothing destroy enough structure to look LESS plausible (positive delta)?")
    print("=" * 88)
    aggressive = per_field_level_diff(rows, "blur", 8, baseline)
    point, lo, hi, n = field_bootstrap_ci(aggressive, args.n_boot, rng)
    excludes_zero = lo > 0 or hi < 0
    print(f"mean delta = {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   n_fields={n}")
    print(f"{'EXCLUDES zero' if excludes_zero else 'includes zero'}"
          + (", direction: INCREASE (less plausible)" if point > 0 else ""))
    print(f"Fraction of fields with positive delta: "
          f"{sum(1 for v in aggressive.values() if v > 0)}/{n}")

    print()
    print("=" * 88)
    print("Compare both CIs against the original single-correlation summary "
          "(rho=0.019, [0.010,0.029]):")
    print("that number was real but averaged two opposite, individually much larger")
    print("effects into a small net figure. Report BOTH phases in the manuscript, not")
    print("the pooled correlation alone.")
    print("=" * 88)


if __name__ == "__main__":
    main()
