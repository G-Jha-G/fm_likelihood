#!/usr/bin/env python3
"""
_review_common.py
====================

Shared statistical helpers for the five review-response scripts
(b1_b2_null_baselines.py, e3_wetfraction_confound_check.py,
rederive_swap_partners_nn.py, table2_fdr_correction.py,
e0_season_stratification_check.py).

Everything here is deliberately copied-with-no-changes from the existing,
already-used machinery in scripts/analyze_e1_bootstrap.py,
scripts/analyze_e1_confound_decomposition.py and
scripts/analyze_e3_conditioning_sensitivity.py (rank correlation, OLS
residuals/R^2, moving block bootstrap, iid field bootstrap), so that the
review-response numbers are computed the SAME way as the manuscript's own
numbers, not by a parallel, unverified reimplementation. The one genuinely
new piece of statistical machinery is bh_fdr() (Benjamini-Hochberg), which
nothing in the existing scripts needed before now (review concern #4).

SEASON CONVENTION
------------------
scripts/compute_e1_covariates.py's own season_of() has an internal naming
bug: it labels months {1, 2} as "DJF" even though December (month 12) is
routed to "OND", so its "DJF" bucket actually only ever contains January
and February. The real e1_analysis_table.csv / e3_selected_fields.csv on
disk use the bucket name "JF" for exactly that same two-month set (see
e3_selected_fields.csv: 2011-01-22 -> season="JF"), so the mislabeling was
evidently caught and fixed downstream of compute_e1_covariates.py itself,
without the source function being corrected. season_of() below reproduces
the CORRECTED four-way split with the labels actually present in the
saved tables (JF, MAM, JJAS, OND), used only as a fallback when a script
needs a season label computed from a date, directly a date and no season
column is available; every script here prefers reading season from a
supplied CSV's own "season" column when one exists, specifically so this
convention question never has to be trusted blindly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# OLS (plain numpy, no extra dependency), matching
# analyze_e1_confound_decomposition.py / analyze_e1_bootstrap.py exactly
# ---------------------------------------------------------------------------

def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coefs
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coefs


def partial_spearman(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    """Partial Spearman of y vs x, controlling for the columns of
    `controls` (already a design matrix WITHOUT intercept; an intercept
    column is added here). Residualized-rank method, identical to
    analyze_e1_confound_decomposition.py's partial_spearman_* functions."""
    ry, rx = rankdata(y), rankdata(x)
    X = np.column_stack([np.ones(len(y)), controls]) if controls.ndim == 2 \
        else np.column_stack([np.ones(len(y)), controls.reshape(-1, 1)])
    y_resid = ols_residuals(ry, X)
    x_resid = ols_residuals(rx, X)
    denom = float(np.sqrt((y_resid ** 2).sum() * (x_resid ** 2).sum()))
    return float((y_resid * x_resid).sum() / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Bootstrap: moving block (time series) and iid (cross-sectional fields)
# ---------------------------------------------------------------------------

def decorrelation_time(series: np.ndarray, threshold: float = 1.0 / np.e,
                        max_lag: int = 90) -> int:
    """First lag at which the ACF drops below `threshold`. Identical to
    analyze_e1_bootstrap.py's decorrelation_time()."""
    n = len(series)
    x = series - series.mean()
    denom = (x ** 2).sum()
    if denom == 0:
        return max_lag
    for lag in range(1, min(max_lag, n - 1) + 1):
        acf = float((x[:-lag] * x[lag:]).sum() / denom)
        if acf < threshold:
            return lag
    return max_lag


def moving_block_bootstrap_sample(n: int, block_length: int, rng: np.random.Generator,
                                   circular: bool = False) -> np.ndarray:
    """circular=False: blocks never run past the end of the series (what
    scripts/analyze_e1_bootstrap.py does). circular=True: block starts are
    drawn from all n positions and blocks wrap around the end (what
    scripts/bootstrap_coverage_power.py and the manuscript's Sect. 3.2 describe)."""
    n_blocks = int(np.ceil(n / block_length))
    if circular:
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block_length)) % n for s in starts])
    else:
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
    return idx[:n]


def block_bootstrap_dist(stat_fn, df: pd.DataFrame, block_length: int, n_boot: int,
                          rng: np.random.Generator, circular: bool = False) -> np.ndarray:
    """df must already be in TIME ORDER. Returns the n_boot bootstrap draws
    of stat_fn(resampled_df), NOT just a CI, so callers can also compute a
    bootstrap p-value from the same draws (table2_fdr_correction.py needs
    exactly that)."""
    n = len(df)
    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = moving_block_bootstrap_sample(n, block_length, rng, circular=circular)
        boot_vals[b] = stat_fn(df.iloc[idx].reset_index(drop=True))
    return boot_vals


def field_bootstrap_dist(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                          stat=np.mean) -> np.ndarray:
    """Ordinary iid bootstrap over independent cross-sectional units
    (E3's 20 fields), NOT over time. Matches
    analyze_e3_conditioning_sensitivity.py's field_bootstrap_ci exactly,
    generalized to an arbitrary summary statistic (default: mean)."""
    n = len(values)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = stat(values[idx])
    return boot


def percentile_ci(boot_vals: np.ndarray, alpha: float = 0.05):
    lo, hi = np.percentile(boot_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_two_sided_pvalue(boot_vals: np.ndarray, null_value: float = 0.0) -> float:
    """Two-sided bootstrap p-value: 2 * min(P(boot <= null), P(boot >= null)),
    capped at 1. This is the standard "proportion on the other side of the
    null" construction (e.g. Davison & Hinkley 1997 Sec 4.2); it needs no
    assumption that the bootstrap distribution is symmetric or Gaussian,
    appropriate here since several of these statistics (partial Spearman
    under a short within-season sample) are not obviously either."""
    n = len(boot_vals)
    p_le = float(np.mean(boot_vals <= null_value))
    p_ge = float(np.mean(boot_vals >= null_value))
    p = 2.0 * min(p_le, p_ge)
    # +1 continuity-style floor so a p-value is never reported as exactly
    # zero purely because no bootstrap draw happened to cross the null;
    # with n_boot draws the smallest resolvable p-value is 1/n_boot.
    p = max(p, 1.0 / n)
    return min(p, 1.0)


# ---------------------------------------------------------------------------
# Multiple comparisons: Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------

def bh_fdr(p_values) -> np.ndarray:
    """Benjamini-Hochberg (1995) step-up FDR correction. Returns q-values
    in the SAME order as the input p_values.

    Procedure: sort p ascending, q_(i) = p_(i) * m / i, then enforce
    monotonicity by taking the running minimum from the largest rank
    down (q_(i) = min(q_(i), q_(i+1))), then clip to [0, 1]. This is the
    textbook step-up form (Benjamini & Hochberg, J. R. Stat. Soc. B,
    1995), verified below against a hand-worked example in
    `_selftest()`.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    ranks = np.empty(m, dtype=np.float64)
    ranks[order] = np.arange(1, m + 1, dtype=np.float64)
    q_sorted = p[order] * m / ranks[order]
    # enforce monotonicity: running minimum from the end
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(m, dtype=np.float64)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def _selftest():
    """Hand-checkable example: p = [0.01, 0.02, 0.03, 0.04, 0.50] (m=5).
    Sorted ranks 1..5, q_(i) = p_(i)*5/i:
      i=1: 0.01*5/1 = 0.05
      i=2: 0.02*5/2 = 0.05
      i=3: 0.03*5/3 = 0.05
      i=4: 0.04*5/4 = 0.05
      i=5: 0.50*5/5 = 0.50
    Already monotone non-decreasing, so q = [0.05, 0.05, 0.05, 0.05, 0.50].
    """
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.50])
    q = bh_fdr(p)
    expected = np.array([0.05, 0.05, 0.05, 0.05, 0.50])
    assert np.allclose(q, expected), f"bh_fdr self-test failed: {q} != {expected}"

    # A case that actually exercises the monotonicity enforcement step:
    # p = [0.04, 0.005], m=2. Sorted: 0.005 (rank1) -> 0.005*2/1=0.01;
    # 0.04 (rank2) -> 0.04*2/2=0.04. Already monotone (0.01 <= 0.04), so
    # q = [0.04, 0.01] in original order.
    p2 = np.array([0.04, 0.005])
    q2 = bh_fdr(p2)
    assert np.allclose(q2, np.array([0.04, 0.01])), q2

    # A case where raw q_(i) is NOT monotone and must be corrected:
    # p = [0.03, 0.9, 0.9, 0.9, 0.9], m=5.
    # sorted: 0.03(r1)->0.15; 0.9(r2)->2.25->clip 1; 0.9(r3)->1.5->clip1;
    # 0.9(r4)->1.125->clip1; 0.9(r5)->0.9. Raw (pre-clip) sorted sequence:
    # [0.15, 2.25, 1.5, 1.125, 0.9]. Running-min from the right over the
    # UNCLIPPED values: [0.15, 0.9, 0.9, 0.9, 0.9] (since min(2.25,1.5,
    # 1.125,0.9)=0.9, and 0.15 stays). Then clip to [0,1]: unchanged.
    p3 = np.array([0.03, 0.9, 0.9, 0.9, 0.9])
    q3 = bh_fdr(p3)
    expected3 = np.array([0.15, 0.9, 0.9, 0.9, 0.9])
    assert np.allclose(q3, expected3), f"{q3} != {expected3}"
    print("bh_fdr self-test: PASS (3/3 hand-worked cases)")


# ---------------------------------------------------------------------------
# Season convention (see module docstring)
# ---------------------------------------------------------------------------

def season_of(date: "pd.Timestamp") -> str:
    m = date.month
    if m in (6, 7, 8, 9):
        return "JJAS"
    if m in (3, 4, 5):
        return "MAM"
    if m in (10, 11, 12):
        return "OND"
    return "JF"  # 1, 2 -- see module docstring re: compute_e1_covariates.py's bug


def test_dates_for_mswx_test_period(n_days: int = 1461) -> "pd.DatetimeIndex":
    """The test period's date index, reproduced identically from
    compute_e1_covariates.py / e3_conditioning_sensitivity.py: a plain
    daily range starting 2011-01-01, field_id == row index into this
    range. 1461 = 365+366+365+365 (2011-2014, correctly spanning the
    2012 leap year)."""
    return pd.date_range("2011-01-01", periods=n_days, freq="D")


if __name__ == "__main__":
    _selftest()
