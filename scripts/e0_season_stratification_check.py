#!/usr/bin/env python3
"""
e0_season_stratification_check.py
=====================================
Review concern #5 (major): E0b/E0c's 6-field solver/probe validation
set (field_ids 2, 48, 134 chosen by wet-fraction extremes; 36, 86, 138
chosen by intensity extremes -- see scripts/e0c_solver_convergence.py's
DEFAULT_FIELD_IDS and scripts/find_extreme_intensity_fields.py) was never
checked for which season each of those 6 days falls in. Sect 4.2's own
finding is that BPD's relationship with extremity is season-dependent
(it may vanish within JJAS, per the brief's own stated expectation,
reproduced in analyze_e1_confound_decomposition.py). If the 6 validation
fields are concentrated in one or two seasons, the numerical-accuracy
conclusions (which RK4 step count / dopri5 tolerance is "good enough")
were only ever actually tested in that slice of the seasonal cycle, and
the manuscript does not currently say so.

WHAT THIS SCRIPT DOES
----------------------
1. Resolves the calendar date of each of the 6 validation field_ids
   (E0b/E0c read the test file with [::10], so field_id k is test day
   10*k, i.e. 2011-01-01 + 10*k days; --stride 1 gives the daily-index
   convention instead, which is the one used by
   compute_e1_covariates.py / e3_conditioning_sensitivity.py) and its
   season under the JF/MAM/JJAS/OND split actually used in the saved
   covariate tables (see _review_common.py's module docstring for why
   this differs from compute_e1_covariates.py's own, differently-named
   internal season_of()).
2. Compares the 6 fields' season counts against the FULL test period's
   season proportions (1461 days, 2011-2014), via a two-sided binomial-
   style check per season using the hypergeometric distribution (6 draws
   without replacement from a population of 1461 with known per-season
   counts) rather than assuming independence across the 6 draws.
3. If --covariates is supplied (e1_covariates.csv or equivalent, with
   columns field_id, season, p999_wet), additionally finds, for every
   season NOT represented (or under-represented) among the 6, the field
   with the highest 99.9th-percentile intensity within that season, as
   concrete candidate field_ids to add to the validation set rather than
   just naming the gap in words. Without --covariates, the script still
   reports the gap, it just can't propose specific replacement field_ids
   by intensity (it can still propose arbitrary ones from within the
   needed season).

CPU-only. No h5 file strictly required for the core season-coverage
check (it only needs a date range and the field_ids), though
--covariates lets it also rank candidate replacement fields by
intensity if that file is available.

USAGE
-----
    python e0_season_stratification_check.py
    python e0_season_stratification_check.py --covariates e1_covariates.csv
    python e0_season_stratification_check.py --field-ids 2,48,134,36,86,138
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_common import season_of, test_dates_for_mswx_test_period  # noqa: E402

DEFAULT_FIELD_IDS = [2, 48, 134, 36, 86, 138]
SELECTION_REASON = {2: "wet-fraction extreme", 48: "wet-fraction extreme",
                     134: "wet-fraction extreme", 36: "intensity extreme",
                     86: "intensity extreme", 138: "intensity extreme"}
SEASONS = ["JF", "MAM", "JJAS", "OND"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field-ids", type=str, default=",".join(str(x) for x in DEFAULT_FIELD_IDS))
    p.add_argument("--stride", type=int, default=10,
                   help="E0b/E0c load the test file with [::10], so their field_id k is test "
                        "day 10*k, not day k. Use --stride 1 only for daily-indexed ids.")
    p.add_argument("--n-days", type=int, default=1461,
                   help="length of the test period (2011-2014 = 1461 days)")
    p.add_argument("--covariates", type=Path, default=None,
                   help="optional e1_covariates.csv (or equivalent) with "
                        "columns field_id, season, p999_wet, used to propose "
                        "concrete replacement/addition field_ids by intensity")
    p.add_argument("--n-propose", type=int, default=2,
                   help="how many candidate fields to propose per under-covered season")
    args = p.parse_args()

    field_ids = [int(x) for x in args.field_ids.split(",")]
    dates = test_dates_for_mswx_test_period(args.n_days)
    if len(dates) != args.n_days:
        print(f"[warn] requested n_days={args.n_days} but got {len(dates)} dates")

    all_seasons = pd.Series([season_of(d) for d in dates])
    pop_counts = all_seasons.value_counts().reindex(SEASONS, fill_value=0)
    pop_total = int(pop_counts.sum())

    print("=" * 88)
    print(f"FULL TEST PERIOD ({pop_total} days, 2011-01-01 + {args.n_days - 1} days):")
    print("=" * 88)
    for s in SEASONS:
        print(f"  {s:>4}: {pop_counts[s]:>4} days  ({100 * pop_counts[s] / pop_total:5.1f}%)")

    rows = []
    for fid in field_ids:
        day = fid * args.stride
        if fid < 0 or day >= len(dates):
            print(f"[warn] field_id={fid} (day {day}) out of range for a {len(dates)}-day period, skipping")
            continue
        d = dates[day]
        rows.append({"field_id": fid, "day_index": day, "date": d.strftime("%Y-%m-%d"),
                      "season": season_of(d),
                      "selection_reason": SELECTION_REASON.get(fid, "(unspecified)")})
    val_df = pd.DataFrame(rows)

    print()
    print("=" * 88)
    print("THE 6 E0b/E0c VALIDATION FIELDS")
    print("=" * 88)
    print(val_df.to_string(index=False))

    val_counts = val_df["season"].value_counts().reindex(SEASONS, fill_value=0)
    n_val = len(val_df)

    print()
    print("=" * 88)
    print("SEASON COVERAGE: observed count among the 6 vs. expected under random draw")
    print("from the full test period's season proportions (hypergeometric, no replacement)")
    print("=" * 88)
    print(f"{'season':>6} {'pop_days':>9} {'pop_frac':>9} {'n_in_6':>7} {'expected':>9} "
          f"{'p_value':>9}  flag")
    any_zero = []
    for s in SEASONS:
        K = int(pop_counts[s])          # "successes" in the population
        k = int(val_counts[s])          # observed successes in the sample of 6
        expected = n_val * K / pop_total
        # Two-sided hypergeometric p-value: sum of P(X=x) over all x at
        # least as extreme (as far from the mean) as the observed k.
        dist = hypergeom(pop_total, K, n_val)
        pmf = np.array([dist.pmf(x) for x in range(n_val + 1)])
        p_obs = dist.pmf(k)
        p_value = float(pmf[pmf <= p_obs + 1e-12].sum())
        flag = ""
        if k == 0 and K > 0:
            flag = "<-- ZERO REPRESENTATION"
            any_zero.append(s)
        elif p_value < 0.10:
            flag = "<-- notably over/under-represented (p<0.10)"
        print(f"{s:>6} {K:>9} {K/pop_total:>8.1%} {k:>7} {expected:>9.2f} {p_value:>9.3f}  {flag}")

    print()
    if any_zero:
        print(f"GAP CONFIRMED: {', '.join(any_zero)} {'is' if len(any_zero)==1 else 'are'} "
              f"entirely unrepresented among the 6 validation fields. The solver/probe "
              f"convergence conclusions (which RK4 step count or dopri5 tolerance is "
              f"'good enough') have not actually been tested on a single day drawn from "
              f"{'this season' if len(any_zero)==1 else 'these seasons'}, and Sect 4.2's own "
              f"finding that the BPD-extremity relationship is season-dependent means this "
              f"is not a safe generalization to assume.")
    else:
        print("No season is at literal zero representation among the 6, though check the "
              "p-value column above for seasons that are still numerically under-sampled.")

    if args.covariates is not None:
        if not args.covariates.exists():
            print(f"\n[warn] --covariates {args.covariates} not found; skipping proposals.")
            return 0
        cov = pd.read_csv(args.covariates)
        needed_cols = {"field_id", "season", "p999_wet"}
        if not needed_cols.issubset(cov.columns):
            print(f"\n[warn] {args.covariates} missing one of {needed_cols}; skipping proposals.")
            return 0

        print()
        print("=" * 88)
        print(f"PROPOSED ADDITIONS (top {args.n_propose} by p999_wet per under-covered season, "
              f"excluding the 6 already selected)")
        print("=" * 88)
        already = set(val_df["day_index"])
        under_covered = [s for s in SEASONS if val_counts[s] == 0 or
                          (n_val * pop_counts[s] / pop_total) - val_counts[s] >= 1.0]
        if not under_covered:
            print("No season flagged as under-covered relative to its population share; "
                  "no additions proposed.")
        for s in under_covered:
            candidates = cov[(cov["season"] == s) & (~cov["field_id"].isin(already))]
            candidates = candidates.sort_values("p999_wet", ascending=False).head(args.n_propose)
            if candidates.empty:
                print(f"  {s}: no candidates found in {args.covariates}")
                continue
            print(f"  {s} (currently {val_counts[s]}/6, population share "
                  f"{pop_counts[s]/pop_total:.1%}):")
            for _, r in candidates.iterrows():
                print(f"      field_id={int(r['field_id']):>5}  p999_wet={r['p999_wet']:.2f} mm "
                      f"<- add to --field-ids for e0b_probe_convergence.py / "
                      f"e0c_solver_convergence.py")

    print()
    print("=" * 88)
    print("NEXT STEP")
    print("=" * 88)
    print("If a gap is confirmed: either (a) re-run e0b_probe_convergence.py and "
          "e0c_solver_convergence.py with --field-ids extended to include the proposed "
          "additions and confirm the same step-count/tolerance choice still passes on "
          "them, or (b) keep the original 6 and add one sentence to the manuscript's "
          "numerical-validation subsection disclosing which season(s) were not covered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
