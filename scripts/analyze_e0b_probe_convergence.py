#!/usr/bin/env python3
"""
analyze_e0b_probe_convergence.py
===================================

The runner script's built-in summary prints a single pooled SD per K,
which mixes between-field and within-field variance and, more importantly,
gives no sense of how UNCERTAIN that SD estimate itself is. At small
sample sizes that uncertainty is not a minor caveat, it can be the whole
story: two draws (n=2) gives a 95% CI on the true SD that can span more
than 100x from lower to upper bound. A small point estimate from 2 draws
is close to uninformative on its own.

This script computes what the brief's decision rule actually needs: for
each (field, K), the sample SD across seeds AND a 95% confidence interval
on the TRUE SD, using the standard chi-squared result for normal samples,

    (n - 1) * s^2 / sigma^2 ~ chi2(n - 1)

which gives an exact interval (no bootstrap needed) as long as the BPD
estimator's noise really is approximately Gaussian, plausible here since
it is a sum of many roughly-independent per-step Hutchinson and
dequantisation contributions, but not verified; treat the interval as a
reasonable approximation, not a certainty.

DECISION RULE: pass a (field, K) only if the UPPER 95% CI bound is below
0.02 bits/dim, not just the point estimate. A point estimate under 0.02
with an upper CI bound above it means the data has not yet ruled out an
unsafe SD, that is a "need more seeds" result, not a pass.

Also reports a POOLED estimate across all fields at each K (more
statistical power, via combined degrees of freedom) alongside the
per-field ones, but flags explicitly that pooling assumes the noise
floor does not depend on field wetness, an assumption this project's own
central thesis (wetness confounds everything about this density) gives
real reason to doubt, so the per-field numbers are the ones to trust for
the actual go/no-go, the pooled number is a supporting sanity check only.

USAGE
-----
    python scripts/analyze_e0b_probe_convergence.py
    python scripts/analyze_e0b_probe_convergence.py --results e0b_outputs/e0b_results.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

TARGET_SD = 0.02


def chi2_ci_on_sd(values: list[float], alpha: float = 0.05):
    """95% (default) CI on the true SD from a normal-noise assumption.
    Returns (sd_hat, ci_lower, ci_upper, df). Returns (sd_hat, nan, nan, 0)
    if n < 2 (cannot estimate spread from a single point)."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"), 0)
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for chi2_ci_on_sd; pip install scipy "
                            "--break-system-packages")
    df = n - 1
    arr = np.asarray(values, dtype=np.float64)
    sd_hat = float(arr.std(ddof=1))
    sum_sq_dev = float(((arr - arr.mean()) ** 2).sum())
    chi2_lo = stats.chi2.ppf(alpha / 2, df)
    chi2_hi = stats.chi2.ppf(1 - alpha / 2, df)
    var_upper = sum_sq_dev / chi2_lo
    var_lower = sum_sq_dev / chi2_hi
    return (sd_hat, float(np.sqrt(var_lower)), float(np.sqrt(var_upper)), df)


def pooled_chi2_ci_on_sd(groups: list[list[float]], alpha: float = 0.05):
    """Pooled variance across several independent groups assumed to share
    a common true SD, standard ANOVA-style pooling: combined df is the
    sum of each group's df, combined sum-of-squares is the sum of each
    group's own sum-of-squared-deviations-from-ITS-OWN-mean (not the grand
    mean), which is what keeps this valid even if the group MEANS differ,
    only the assumption that the SPREAD around each mean is common."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required; pip install scipy --break-system-packages")
    usable = [g for g in groups if len(g) >= 2]
    if not usable:
        return (float("nan"), float("nan"), float("nan"), 0)
    combined_df = sum(len(g) - 1 for g in usable)
    combined_sum_sq = sum(float(((np.asarray(g) - np.mean(g)) ** 2).sum()) for g in usable)
    pooled_sd_hat = float(np.sqrt(combined_sum_sq / combined_df))
    chi2_lo = stats.chi2.ppf(alpha / 2, combined_df)
    chi2_hi = stats.chi2.ppf(1 - alpha / 2, combined_df)
    var_upper = combined_sum_sq / chi2_lo
    var_lower = combined_sum_sq / chi2_hi
    return (pooled_sd_hat, float(np.sqrt(var_lower)), float(np.sqrt(var_upper)), combined_df)


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("e0b_outputs/e0b_results.jsonl"))
    p.add_argument("--target-sd", type=float, default=TARGET_SD)
    args = p.parse_args()

    if not args.results.exists():
        print(f"error: {args.results} not found")
        return 1
    if not _HAS_SCIPY:
        print("error: this script needs scipy. pip install scipy --break-system-packages")
        return 1

    rows = load_results(args.results)
    print(f"Loaded {len(rows)} results from {args.results}\n")

    by_field_k = defaultdict(list)
    for r in rows:
        by_field_k[(r["field_id"], r["K"])].append(r["bits_per_interior_dim"])

    fields = sorted({fid for fid, _ in by_field_k})
    ks = sorted({K for _, K in by_field_k})

    print("=" * 88)
    print(f"PER-FIELD, PER-K: sample SD and 95% CI on the true SD (target: < {args.target_sd})")
    print("=" * 88)
    print(f"{'field':>6} {'K':>4} {'n':>3} {'df':>3} {'SD_hat':>9} "
          f"{'CI_lower':>10} {'CI_upper':>10}  verdict")
    print("-" * 88)
    any_pass_per_field = {}
    for field in fields:
        for K in ks:
            vals = by_field_k.get((field, K))
            if not vals:
                continue
            sd_hat, lo, hi, df = chi2_ci_on_sd(vals)
            if df == 0:
                verdict = "n=1, cannot estimate SD"
            elif hi < args.target_sd:
                verdict = "PASS (CI upper clears target)"
                any_pass_per_field.setdefault(field, set()).add(K)
            else:
                verdict = "NOT ESTABLISHED (CI upper exceeds target, need more seeds)"
            sd_str = f"{sd_hat:.5f}" if df > 0 else "  n/a  "
            lo_str = f"{lo:.5f}" if df > 0 else "  n/a  "
            hi_str = f"{hi:.5f}" if df > 0 else "  n/a  "
            print(f"{field:>6} {K:>4} {len(vals):>3} {df:>3} {sd_str:>9} "
                  f"{lo_str:>10} {hi_str:>10}  {verdict}")
    print()

    print("=" * 88)
    print("POOLED ACROSS FIELDS, PER K (supporting check only, assumes common noise SD")
    print("across fields, an assumption this project's own thesis gives reason to doubt)")
    print("=" * 88)
    print(f"{'K':>4} {'df':>4} {'pooled_SD':>10} {'CI_lower':>10} {'CI_upper':>10}  verdict")
    print("-" * 88)
    for K in ks:
        groups = [by_field_k[(f, K)] for f in fields if (f, K) in by_field_k]
        sd_hat, lo, hi, df = pooled_chi2_ci_on_sd(groups)
        if df == 0:
            continue
        verdict = "PASS" if hi < args.target_sd else "NOT ESTABLISHED"
        print(f"{K:>4} {df:>4} {sd_hat:>10.5f} {lo:>10.5f} {hi:>10.5f}  {verdict}")
    print()

    print("=" * 88)
    print("RECOMMENDATION")
    print("=" * 88)
    fields_with_pass = set(any_pass_per_field.keys())
    all_fields = set(fields)
    if fields_with_pass == all_fields and all_fields:
        smallest_passing_k_per_field = {f: min(any_pass_per_field[f]) for f in fields}
        overall_k = max(smallest_passing_k_per_field.values())
        print(f"Every field with data has at least one K that PASSES (CI upper < target). "
              f"Smallest K that passes on every field so far: {overall_k}.")
        print(f"If that holds up, K={overall_k} would be the answer to the brief's question, "
              f"and testing higher K (up to the spec's 64) is likely unnecessary.")
    else:
        missing = all_fields - fields_with_pass
        print(f"Fields with NO K yet established as passing: {sorted(missing) if missing else 'none'}")
        print("For any field still short of a PASS at the K values tested, the fix is more "
              "SEEDS at the cheapest untested or borderline K, not more probes. K=1 costs "
              "~123s/unit versus ~4000s/unit at K=64 (measured directly from your pilot log), "
              "so it is far cheaper to narrow the CI at low K than to jump to a higher K on "
              "the assumption that more probes must mean less noise, that assumption needs "
              "checking too, not assumed.")
    print()
    print("Note on n=1 rows above (if any): a single draw gives no information about spread "
          "at all, df=0, no CI possible. Those need at least a second seed before they say "
          "anything.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
