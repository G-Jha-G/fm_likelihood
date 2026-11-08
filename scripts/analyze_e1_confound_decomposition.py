#!/usr/bin/env python3
"""
analyze_e1_confound_decomposition.py
=======================================

E1's core question (brief Section E1): how much of the between-field
variance in BPD is explained by wet fraction, domain mean intensity,
season, and spatial organisation, IN THAT ORDER, and does BPD still
correlate with extremity once wet fraction and season are controlled?

This covers the first two of E1's five specified elements:
  1. Hierarchical variance partitioning, incremental R^2 over nested
     models, covariates added in the brief's exact order
  2. Partial Spearman correlation between BPD and extremity (p999_wet),
     controlling for wet fraction and season, plus within-season
     Spearman reported separately

NOT yet covered here (separate script, since they're different
statistical machinery and this experiment is worth not rushing):
  3. Moving block bootstrap for valid uncertainty (this script's p-values
     and R^2 are NOT yet corrected for the temporal autocorrelation N7
     flagged; treat them as a first, uncorrected look)
  4. Matched-pair analysis (confound-free version of the pilot's
     Analysis B)

COVARIATE GROUPS, IN THE BRIEF'S SPECIFIED ORDER
---------------------------------------------------
  1. wet fraction:          dry_fraction, wet_fraction
  2. domain mean intensity: domain_mean, wet_day_mean, p999_wet, spatial_max
  3. season:                season (dummy-coded), day_of_year
  4. spatial organisation:  raps_log_slope, n_connected_wet_components,
                             mean_component_size,
                             intensity_concentration_top1pct, morans_i

Note p999_wet sits in group 2 (domain mean intensity) here, but is ALSO
the "extremity" variable in the partial-correlation test. That's
intentional, not a conflict: the R^2 decomposition asks how much of
BPD's variance intensity explains overall; the partial correlation asks,
of the extremity-BPD relationship specifically, how much survives once
wet fraction (group 1) and season (part of group 3) are held fixed. Two
different questions using the same variable in different roles.

STATISTICS
----------
Partial Spearman via residualized ranks: rank-transform BPD and
extremity, regress each on rank(wet_fraction) + season dummies via OLS,
correlate the residuals. Hierarchical R^2 via OLS on RAW (not ranked)
covariates, since R^2 partitioning conventionally uses the natural
variable scale, not ranks; this uses one convention for one question,
consistently, not inconsistently.

WITHIN-SEASON: partial (controlling for wet_fraction), not raw. A raw
within-season correlation still picks up the wet_fraction confound
equally in every season, and a synthetic test with an effect planted
only in one season confirmed the raw version fails to localize it while
the partial version does. Matches the brief's own phrasing ("partial
rho... may vanish within JJAS", describing the partial statistic).

USAGE
-----
    python scripts/analyze_e1_confound_decomposition.py
    python scripts/analyze_e1_confound_decomposition.py --table e1_analysis_table.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """R^2 of y regressed on X (X should already include an intercept
    column of ones). Plain numpy lstsq, no extra dependency."""
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coefs
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coefs


def _design_matrix(df: pd.DataFrame, columns: list[str], season_dummies: bool = False
                    ) -> np.ndarray:
    """Builds an OLS design matrix (with intercept) from a list of
    dataframe columns. 'season' gets dummy-coded (drop first level to
    avoid collinearity with the intercept) if season_dummies is True and
    'season' is in columns; every other column is used as-is."""
    parts = [np.ones((len(df), 1))]
    for c in columns:
        if c == "season" and season_dummies:
            dummies = pd.get_dummies(df["season"], drop_first=True)
            parts.append(dummies.to_numpy(dtype=np.float64))
        elif c == "season":
            continue  # skip raw season string if not dummy-coding
        else:
            parts.append(df[[c]].to_numpy(dtype=np.float64))
    return np.hstack(parts)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


GROUP1 = ["dry_fraction", "wet_fraction"]
GROUP2 = ["domain_mean", "wet_day_mean", "p999_wet", "spatial_max"]
GROUP3 = ["season", "day_of_year"]
GROUP4 = ["raps_log_slope", "n_connected_wet_components", "mean_component_size",
          "intensity_concentration_top1pct", "morans_i"]
EXTREMITY_COL = "p999_wet"


def hierarchical_variance_partition(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    y = df[y_col].to_numpy(dtype=np.float64)
    cumulative_cols: list[str] = []
    rows = []
    prev_r2 = 0.0
    null_X = np.ones((len(df), 1))
    null_r2 = ols_r2(y, null_X)
    rows.append({"model": "null (intercept only)", "added_group": "-",
                 "cumulative_R2": null_r2, "incremental_R2": null_r2})
    prev_r2 = null_r2

    for group_name, group_cols in [("1: wet fraction", GROUP1),
                                    ("2: domain mean intensity", GROUP2),
                                    ("3: season", GROUP3),
                                    ("4: spatial organisation", GROUP4)]:
        cumulative_cols += group_cols
        X = _design_matrix(df, cumulative_cols, season_dummies=True)
        r2 = ols_r2(y, X)
        rows.append({"model": f"+ group {group_name}", "added_group": group_name,
                      "cumulative_R2": r2, "incremental_R2": r2 - prev_r2})
        prev_r2 = r2

    return pd.DataFrame(rows)


def partial_spearman_extremity_given_wetness_and_season(df: pd.DataFrame, y_col: str
                                                          ) -> dict:
    y = df[y_col].to_numpy(dtype=np.float64)
    ext = df[EXTREMITY_COL].to_numpy(dtype=np.float64)

    raw_rho = _spearman(y, ext)

    ry = _rankdata(y)
    rext = _rankdata(ext)

    control_X = _design_matrix(df, ["wet_fraction", "season"], season_dummies=True)
    y_resid = ols_residuals(ry, control_X)
    ext_resid = ols_residuals(rext, control_X)
    denom = float(np.sqrt((y_resid ** 2).sum() * (ext_resid ** 2).sum()))
    partial_rho = float((y_resid * ext_resid).sum() / denom) if denom > 0 else float("nan")

    return {"raw_spearman_bpd_vs_extremity": raw_rho,
            "partial_spearman_given_wet_fraction_and_season": partial_rho,
            "n": len(df)}


def within_season_spearman(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Partial Spearman (BPD vs extremity, controlling for wet_fraction),
    computed SEPARATELY within each season subset. Season itself isn't a
    control variable here since it's fixed by construction within each
    subgroup. Deliberately NOT the raw within-season correlation: a raw
    correlation still picks up the wet_fraction confound in every season
    equally, and would not localize a season-specific extremity effect
    even if one exists, confirmed directly by a synthetic test where an
    effect planted only in JJAS failed to show up in the raw version and
    was correctly localized once this used the partial version instead.
    This matches the brief's own phrasing: "partial rho... may vanish
    within JJAS", describing the partial statistic, not the raw one."""
    rows = []
    for season, sub in df.groupby("season"):
        y = sub[y_col].to_numpy(dtype=np.float64)
        ext = sub[EXTREMITY_COL].to_numpy(dtype=np.float64)
        wf = sub["wet_fraction"].to_numpy(dtype=np.float64)

        raw_rho = _spearman(y, ext)

        ry, rext, rwf = _rankdata(y), _rankdata(ext), _rankdata(wf)
        X = np.column_stack([np.ones(len(sub)), rwf])
        y_resid = ols_residuals(ry, X)
        ext_resid = ols_residuals(rext, X)
        denom = float(np.sqrt((y_resid ** 2).sum() * (ext_resid ** 2).sum()))
        partial_rho = float((y_resid * ext_resid).sum() / denom) if denom > 0 else float("nan")

        rows.append({"season": season, "n": len(sub),
                      "raw_spearman": raw_rho,
                      "partial_spearman_given_wet_fraction": partial_rho})
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=Path, default=Path("e1_analysis_table.csv"))
    p.add_argument("--y-col", type=str, default="bits_per_interior_dim",
                   help="which BPD column to use as the dependent variable")
    args = p.parse_args()

    if not args.table.exists():
        print(f"error: {args.table} not found")
        return 1

    df = pd.read_csv(args.table)
    print(f"Loaded {len(df)} rows from {args.table}")
    print(f"Dependent variable: {args.y_col}\n")

    print("=" * 88)
    print("HIERARCHICAL VARIANCE PARTITIONING")
    print("Question: how much of BPD's between-field variance does each covariate group")
    print("explain, added in the brief's specified order?")
    print("=" * 88)
    hvp = hierarchical_variance_partition(df, args.y_col)
    print(hvp.to_string(index=False))
    print()

    print("=" * 88)
    print(f"PARTIAL SPEARMAN: BPD vs extremity ({EXTREMITY_COL}), controlling for "
          f"wet_fraction and season")
    print("=" * 88)
    partial = partial_spearman_extremity_given_wetness_and_season(df, args.y_col)
    for k, v in partial.items():
        print(f"  {k}: {v}")
    drop_pct = (1 - abs(partial["partial_spearman_given_wet_fraction_and_season"])
                / abs(partial["raw_spearman_bpd_vs_extremity"])) * 100
    print(f"\n  Drop in |rho| after controlling for wet fraction and season: {drop_pct:.1f}%")
    print()

    print("=" * 88)
    print("WITHIN-SEASON SPEARMAN (BPD vs extremity, no other controls)")
    print("=" * 88)
    ws = within_season_spearman(df, args.y_col)
    print(ws.to_string(index=False))
    print()

    print("=" * 88)
    print("READING GUIDE (per the brief's stated expectation)")
    print("=" * 88)
    print("Expected: wet fraction and dry fraction explain the majority of the variance,")
    print("and the partial correlation drops substantially, possibly vanishing within JJAS.")
    print("Both outcomes are informative here, there is no go/no-go, only which paper this")
    print("becomes. This script does NOT yet correct for temporal autocorrelation (N7); the")
    print("R^2 and correlation VALUES above are informative, the apparent PRECISION is not")
    print("yet trustworthy until the block bootstrap (next script) is run on top of this.")


if __name__ == "__main__":
    main()
