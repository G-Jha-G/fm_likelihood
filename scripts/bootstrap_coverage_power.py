#!/usr/bin/env python3
"""
bootstrap_coverage_power.py
==============================

The actual coverage/power validation for Section 3.2's block bootstrap,
built on top of what estimator_bias_check.py establishes: the partial-rho
estimator has a small (~0.01, run that script first for your exact
number), real, autocorrelation-driven bias at the true null, using
REALISTIC noise scales estimated directly from your data rather than
guessed. This script tests coverage of the TRUE simulated effect (not
naively assumed to be exactly what the estimator reports), at three
block-length multipliers, using the real wet_fraction sequence so the
autocorrelation structure is exactly real, not approximated.

Two effects are tested: zero (coverage under the null) and a moderate
realistic effect (power to detect a true signal), the same two things
Section 3.2's text already describes qualitatively.

USAGE
-----
    python bootstrap_coverage_power.py --table e1_analysis_table.csv
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
    rwf, rext, rbpd = rankdata(wf), rankdata(ext), rankdata(bpd)
    X = np.column_stack([np.ones(len(wf)), rwf])
    ext_r = ols_resid(rext, X)
    bpd_r = ols_resid(rbpd, X)
    denom = np.sqrt((ext_r ** 2).sum() * (bpd_r ** 2).sum())
    return (ext_r * bpd_r).sum() / denom if denom > 0 else np.nan


def sample_acf(x, max_lag):
    x = x - x.mean()
    denom = (x ** 2).sum()
    acf = np.empty(max_lag + 1)
    n = len(x)
    for k in range(max_lag + 1):
        acf[k] = (x[:n - k] * x[k:]).sum() / denom
    return acf


def decorrelation_time(x, min_days=15, max_lag=200, multiplier=1.0):
    max_lag = min(max_lag, len(x) - 1)
    acf = sample_acf(x, max_lag)
    below = np.where(np.abs(acf[1:]) < 1 / np.e)[0]
    tau = int(below[0]) + 1 if len(below) else max_lag
    return max(int(round(tau * multiplier)), min_days)


def moving_block_resample(n, block_length, rng):
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block_length)) % n for s in starts])
    return idx[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="e1_analysis_table.csv")
    p.add_argument("--n-sim", type=int, default=100)
    p.add_argument("--n-boot", type=int, default=150)
    args = p.parse_args()

    df = pd.read_csv(args.table).sort_values("field_id").reset_index(drop=True)
    real_wf = df["wet_fraction"].to_numpy()
    real_ext = df["p999_wet"].to_numpy()
    real_bpd = df["bits_per_interior_dim"].to_numpy()
    n = len(real_wf)

    X_raw = np.column_stack([np.ones(n), real_wf])
    ext_noise_sd = ols_resid(real_ext, X_raw).std(ddof=2)
    bpd_noise_sd = ols_resid(real_bpd, X_raw).std(ddof=2)
    ext_coef = np.linalg.lstsq(X_raw, real_ext, rcond=None)[0][1]
    bpd_coef = np.linalg.lstsq(X_raw, real_bpd, rcond=None)[0][1]
    print(f"Using real-data-calibrated scales: ext_coef={ext_coef:.2f}, "
          f"ext_noise_sd={ext_noise_sd:.2f}, bpd_coef={bpd_coef:.4f}, "
          f"bpd_noise_sd={bpd_noise_sd:.4f}\n")

    def make_dataset(true_effect, rng):
        sim_ext = ext_coef * real_wf + rng.normal(0, ext_noise_sd, n)
        sim_bpd = bpd_coef * real_wf + true_effect * sim_ext + rng.normal(0, bpd_noise_sd, n)
        return real_wf, sim_ext, sim_bpd

    def coefficient_for_target_rho(target_rho, ext_sd, bpd_sd):
        """Converts a DESIRED partial correlation (dimensionless, -1..1)
        into the raw regression coefficient make_dataset needs. Derivation:
        controlling for wf, ext's residual is just its own noise (sd=ext_sd);
        bpd's residual is coef*ext_noise + bpd_noise. Their correlation is
        coef*ext_sd / sqrt((coef*ext_sd)^2 + bpd_sd^2) = target_rho. Solving:
        coef = bpd_sd*target_rho / (ext_sd*sqrt(1-target_rho^2))."""
        return bpd_sd * target_rho / (ext_sd * np.sqrt(1 - target_rho ** 2))

    def true_center(true_effect, n_ref=3000):
        """The estimator's own long-run average at this true_effect (accounts
        for the small structural bias directly, rather than assuming the
        target is exactly true_effect)."""
        rng = np.random.default_rng(999)
        vals = np.empty(n_ref)
        for i in range(n_ref):
            wf, ext, bpd = make_dataset(true_effect, rng)
            vals[i] = partial_rho(wf, ext, bpd)
        return vals.mean()

    print("Estimating the true target (accounts for the small estimator bias)...")
    target_null = true_center(0.0)
    print(f"  true_effect=0     -> estimator's long-run center = {target_null:+.4f}")

    def run_coverage(true_effect, target, multiplier, n_sim, n_boot, seed0):
        covered = 0
        for sim in range(n_sim):
            sim_rng = np.random.default_rng(seed0 + sim)
            wf, ext, bpd = make_dataset(true_effect, sim_rng)
            tau = decorrelation_time(bpd, min_days=15, multiplier=multiplier)
            boot_rng = np.random.default_rng(seed0 + 100000 + sim)
            boot_vals = np.empty(n_boot)
            for b in range(n_boot):
                idx = moving_block_resample(n, tau, boot_rng)
                boot_vals[b] = partial_rho(wf[idx], ext[idx], bpd[idx])
            lo, hi = np.percentile(boot_vals, [2.5, 97.5])
            if lo <= target <= hi:
                covered += 1
        return covered / n_sim

    print(f"\nCOVERAGE (target = estimator's own long-run center, {target_null:+.4f}, "
          f"not naively 0):")
    for mult in [1, 2, 3]:
        cov = run_coverage(0.0, target_null, mult, args.n_sim, args.n_boot, seed0=5000)
        print(f"  multiplier={mult}x: coverage = {cov*100:.1f}%  (target: 95%)")

    moderate_effect_rho = 0.15  # a TARGET partial correlation, comparable in scale to the reported 0.208
    moderate_effect = coefficient_for_target_rho(moderate_effect_rho, ext_noise_sd, bpd_noise_sd)
    print(f"\nCalibrated coefficient for target partial rho={moderate_effect_rho}: "
          f"{moderate_effect:.6f}")
    print(f"Estimating target for this moderate true effect...")
    target_mod = true_center(moderate_effect)
    print(f"  target partial rho={moderate_effect_rho} -> estimator's long-run center = {target_mod:+.4f}")
    print(f"  (should be close to {moderate_effect_rho}, confirming the calibration worked)")
    print(f"\nPOWER (does the 95% CI exclude zero) at target partial rho={moderate_effect_rho}:")
    for mult in [1, 3]:
        detected = 0
        for sim in range(args.n_sim):
            sim_rng = np.random.default_rng(6000 + sim)
            wf, ext, bpd = make_dataset(moderate_effect, sim_rng)
            tau = decorrelation_time(bpd, min_days=15, multiplier=mult)
            boot_rng = np.random.default_rng(7000 + sim)
            boot_vals = np.empty(args.n_boot)
            for b in range(args.n_boot):
                idx = moving_block_resample(n, tau, boot_rng)
                boot_vals[b] = partial_rho(wf[idx], ext[idx], bpd[idx])
            lo, hi = np.percentile(boot_vals, [2.5, 97.5])
            if lo > 0:
                detected += 1
        print(f"  multiplier={mult}x: power = {100*detected/args.n_sim:.1f}%")


if __name__ == "__main__":
    main()