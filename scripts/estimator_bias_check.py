#!/usr/bin/env python3
"""
estimator_bias_check.py
==========================

Checks something more fundamental than bootstrap coverage: is the
rank-based partial-correlation estimator ITSELF biased under the real
data's actual autocorrelation strength, before any bootstrap is even
involved?

WHY THIS MATTERS: quick synthetic testing (documented in chat, not
included here) found that this estimator shows a small but statistically
unambiguous bias at true partial correlation = 0, and the bias grows
with how strongly autocorrelated the control variable (wet fraction) is.
Real wet_fraction is strongly autocorrelated (annual cycle). If this bias
is present at a magnitude comparable to E1's reported partial correlation
(0.208), that would mean the headline number itself, not just its
confidence interval, needs a closer look.

METHOD: estimate the REAL residual noise scale for extremity and BPD
(after removing their wet-fraction dependence) directly from your actual
e1_analysis_table.csv, rather than guessing. Then simulate synthetic
extremity and BPD built from the REAL wet_fraction sequence (so it has
the real, exact autocorrelation structure) plus freshly-drawn noise at
the ESTIMATED real scale, with a KNOWN true partial effect of exactly
zero. Compute the partial-rho estimator on many such simulated datasets
and check how far its average value sits from the true zero.

USAGE
-----
    python estimator_bias_check.py --table e1_analysis_table.csv
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def ols_resid(y, X):
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coefs


def partial_rho(wf, ext, bpd):
    """Same method as the real analysis: rank both variables, regress
    each on ranked wet fraction, correlate residuals."""
    rwf, rext, rbpd = rankdata(wf), rankdata(ext), rankdata(bpd)
    X = np.column_stack([np.ones(len(wf)), rwf])
    ext_r = ols_resid(rext, X)
    bpd_r = ols_resid(rbpd, X)
    denom = np.sqrt((ext_r ** 2).sum() * (bpd_r ** 2).sum())
    return (ext_r * bpd_r).sum() / denom if denom > 0 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="e1_analysis_table.csv")
    p.add_argument("--n-sim", type=int, default=200)
    args = p.parse_args()

    df = pd.read_csv(args.table).sort_values("field_id").reset_index(drop=True)
    real_wf = df["wet_fraction"].to_numpy()
    real_ext = df["p999_wet"].to_numpy()
    real_bpd = df["bits_per_interior_dim"].to_numpy()
    n = len(real_wf)

    # Estimate the REAL residual noise scale after removing wet-fraction
    # dependence, on the RAW (not ranked) scale, so the synthetic data's
    # noise level matches the actual data as closely as possible.
    X_raw = np.column_stack([np.ones(n), real_wf])
    ext_resid_raw = ols_resid(real_ext, X_raw)
    bpd_resid_raw = ols_resid(real_bpd, X_raw)
    ext_noise_sd = ext_resid_raw.std(ddof=2)
    bpd_noise_sd = bpd_resid_raw.std(ddof=2)
    ext_coef = np.linalg.lstsq(X_raw, real_ext, rcond=None)[0][1]
    bpd_coef = np.linalg.lstsq(X_raw, real_bpd, rcond=None)[0][1]

    print(f"n = {n}")
    print(f"Estimated real relationship: extremity ~ {ext_coef:.2f} * wet_fraction "
          f"+ noise(sd={ext_noise_sd:.3f})")
    print(f"Estimated real relationship: BPD ~ {bpd_coef:.4f} * wet_fraction "
          f"+ noise(sd={bpd_noise_sd:.4f})")
    print(f"(these are the REAL noise scales this test uses, not guesses)\n")

    print(f"Real, actual reported partial correlation on your data: "
          f"{partial_rho(real_wf, real_ext, real_bpd):.4f}")
    print(f"(for reference only; the number below is what matters for the bias check)\n")

    # Now: simulate at TRUE partial effect = 0 exactly, using the REAL wf
    # sequence (so autocorrelation is exactly real) and REAL-scale noise,
    # repeated many times with fresh noise draws each time.
    rng = np.random.default_rng(0)
    points = np.empty(args.n_sim)
    for i in range(args.n_sim):
        sim_ext = ext_coef * real_wf + rng.normal(0, ext_noise_sd, n)
        sim_bpd = bpd_coef * real_wf + rng.normal(0, bpd_noise_sd, n)
        points[i] = partial_rho(real_wf, sim_ext, sim_bpd)

    mean_bias = points.mean()
    se = points.std(ddof=1) / np.sqrt(args.n_sim)
    t_stat = mean_bias / se

    print("=" * 78)
    print(f"ESTIMATOR BIAS CHECK: true partial effect is EXACTLY ZERO by construction.")
    print(f"n_sim={args.n_sim} independent noise draws, real wf sequence held fixed.")
    print("=" * 78)
    print(f"Mean estimated partial rho (should be ~0 if unbiased): {mean_bias:+.4f}")
    print(f"Std across simulations: {points.std(ddof=1):.4f}")
    print(f"Standard error of the mean: {se:.4f}")
    print(f"t-statistic vs zero: {t_stat:+.2f}")
    print()
    print(f"For comparison, E1's actual reported partial correlation is 0.208.")
    print(f"If the bias above is a substantial fraction of 0.208, that is a real")
    print(f"concern for the headline number, not just its confidence interval.")
    print(f"If it's small relative to 0.208 (say, under ~10-15% of it), the bias")
    print(f"is real but not large enough to materially change the paper's claim.")


if __name__ == "__main__":
    main()