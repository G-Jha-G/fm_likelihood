#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e0a_replicate_spread.py
========================

Phase 1a / Experiment E0a: a free numerical-noise-floor estimate from the
pilot's already-saved output.

WHY THIS EXISTS
---------------
fm_likelihood_extremes_v2.py shares one stateful Hutchinson generator
(engine._gen) and one stateful dequantisation generator (dq) across every
analysis inside run_full. Because of that, four arrays already sitting in
likelihood_extremes.npz are four independent stochastic realisations of the
SAME quantity: the conditional BPD of the same 147 real, unperturbed fields
under their own true conditioning.

    lve_bpd       Analysis B, likelihood vs extremity
    typ_real      Analysis C, spatial-shuffle test, "observed" half
    rvr_bpd       Analysis D, rarity vs field maximum
    cmm_matched   Analysis E, conditioning-mismatch test, "matched" half

Their spread, field by field, is a zero-additional-compute measurement of
the combined Hutchinson-probe plus dequantisation noise floor on the real
trained network, not on a toy problem. This is PROJECT_BRIEF Section 14,
"the one thing to do first": it decides how much of the pilot's headline
correlations (rho = 0.585 vs field intensity, rho = 0.669 vs field maximum)
can be trusted before a single new experiment is run.

DECISION RULE (brief Section 14 / research programme finding N4)
------------------------------------------------------------------
    replicate SD < 0.05 bits/dim   numerical noise negligible against the
                                    ~0.62 bits/dim between-field SD; pilot
                                    effects survive; report as Figure 1e
    0.05 to 0.2 bits/dim           correlations are attenuated, must be
                                    disattenuated or carry an explicit
                                    numerical error bar; claim D5 (the
                                    return-level panel giving a "cleaner
                                    correspondence") is dead either way
    > 0.2 bits/dim                 the extremity correlation is unsafe at
                                    8 Hutchinson probes; Analysis B must be
                                    rerun at a higher probe count (E0b)
                                    before anything else proceeds

USAGE
-----
    python e0a_replicate_spread.py --npz /path/to/likelihood_extremes.npz

No GPU, no torch, no h5py. Just numpy, plus matplotlib if you want the
figure. Safe to run on a laptop; the heavy compute already happened when
the pilot produced the npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPLICATE_KEYS = ["lve_bpd", "typ_real", "rvr_bpd", "cmm_matched"]
BETWEEN_FIELD_SD_REPORTED = 0.6235  # from the pilot JSON, sanity cross-check only


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


def extract_replicates(data: np.lib.npyio.NpzFile) -> dict:
    missing = [k for k in REPLICATE_KEYS if k not in data.files]
    if missing:
        raise KeyError(
            f"npz is missing expected keys {missing}. Found: {sorted(data.files)}. "
            "This script expects the arrays exactly as written by "
            "fm_likelihood_extremes_v2.py's save_results()."
        )
    arrays = {k: np.asarray(data[k], dtype=np.float64) for k in REPLICATE_KEYS}
    lengths = {k: v.shape[0] for k, v in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Replicate arrays have mismatched lengths: {lengths}. They must be "
            "field-aligned (same 147 fields, same order) for a paired spread "
            "estimate to be valid."
        )
    return arrays


def variance_decomposition(arrays: dict) -> dict:
    """Between-field vs within-field (numerical) variance, ANOVA style."""
    M = np.stack([arrays[k] for k in REPLICATE_KEYS], axis=1)  # (n_fields, 4)
    n_fields, n_rep = M.shape

    field_mean = M.mean(axis=1)
    between_field_var = float(np.var(field_mean, ddof=1))
    per_field_var = M.var(axis=1, ddof=1)
    within_field_var = float(per_field_var.mean())
    total_var = float(np.var(M.reshape(-1), ddof=1))

    return {
        "n_fields": int(n_fields),
        "n_replicates": int(n_rep),
        "between_field_sd": float(np.sqrt(between_field_var)),
        "within_field_sd_numerical_noise": float(np.sqrt(within_field_var)),
        "total_sd_pooled": float(np.sqrt(total_var)),
        "per_field_replicate_sd": per_field_var ** 0.5,
        "field_mean_bpd": field_mean,
        "raw_matrix": M,
    }


def pairwise_summary(arrays: dict) -> dict:
    out = {}
    keys = REPLICATE_KEYS
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = arrays[keys[i]] - arrays[keys[j]]
            out[f"{keys[i]}_minus_{keys[j]}"] = {
                "mean_diff": float(d.mean()),
                "sd_diff": float(d.std(ddof=1)),
                "mean_abs_diff": float(np.abs(d).mean()),
                "max_abs_diff": float(np.abs(d).max()),
            }
    return out


def correlation_spread(data: np.lib.npyio.NpzFile, arrays: dict) -> dict:
    """How much does rho(covariate, BPD) shift depending purely on which of
    the 4 noisy BPD realisations is used. Directly tests claim D5: the
    manuscript's "return level" panel reports rho = 0.669 against the
    extremity panel's rho = 0.585 as if that gap were a cleaner physical
    correspondence. Part of it may just be which realisation landed where.
    """
    out = {}
    if "lve_extremity" in data.files:
        extremity = np.asarray(data["lve_extremity"], dtype=np.float64)
        out["rho_vs_extremity_99p9"] = {k: _spearman(extremity, v) for k, v in arrays.items()}
    if "rvr_rl" in data.files:
        field_max = np.asarray(data["rvr_rl"], dtype=np.float64)
        out["rho_vs_field_max"] = {k: _spearman(field_max, v) for k, v in arrays.items()}
    return out


def verdict(numerical_sd: float) -> str:
    if numerical_sd < 0.05:
        return (
            "PASS. Numerical noise is negligible against the between-field spread. "
            "The pilot's headline correlations are not an artifact of Hutchinson or "
            "dequantisation noise. Report this as Figure 1e and proceed to E0b-g for "
            "the full numerical budget, then E1."
        )
    if numerical_sd <= 0.20:
        return (
            "CAUTION. Correlations are attenuated by numerical noise and must be "
            "disattenuated, or reported with an explicit numerical error bar. D5 "
            "(the claim that the return-level panel gives a cleaner correspondence "
            "than the extremity panel) is dead: part of the 0.585 vs 0.669 gap is "
            "this noise floor, not signal. Proceed to E0b (probe convergence) before "
            "trusting any single correlation value."
        )
    return (
        "STOP. Numerical noise exceeds the safe threshold. Analysis B is unsafe at "
        "8 Hutchinson probes. Do not run E1 or anything downstream until E0b (probe "
        "convergence) identifies a probe count that brings this below 0.20 bits/dim, "
        "ideally below 0.05."
    )


def make_figure(decomp: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable ({e}); skipping figure.", file=sys.stderr)
        return

    M = decomp["raw_matrix"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax0 = axes[0]
    ax0.scatter(M[:, 0], M[:, 2], s=14, alpha=0.6, edgecolor="none")
    lo = min(M[:, [0, 2]].min(), 0.0)
    hi = M[:, [0, 2]].max() * 1.05
    ax0.plot([lo, hi], [lo, hi], color="0.5", lw=1, ls="--")
    ax0.set_xlabel(f"{REPLICATE_KEYS[0]} (bits/dim)")
    ax0.set_ylabel(f"{REPLICATE_KEYS[2]} (bits/dim)")
    ax0.set_title("Two realisations of the same 147 BPDs")

    ax1 = axes[1]
    ax1.hist(decomp["per_field_replicate_sd"], bins=30, color="#4C72B0", alpha=0.85)
    ax1.axvline(0.05, color="green", ls="--", lw=1, label="0.05 (safe)")
    ax1.axvline(0.20, color="red", ls="--", lw=1, label="0.20 (unsafe)")
    ax1.set_xlabel("per-field replicate SD (bits/dim)")
    ax1.set_ylabel("count")
    ax1.set_title("Numerical noise floor, per field")
    ax1.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--npz", type=Path,
        default=Path("./likelihood_outputs/likelihood_extremes.npz"),
        help="Path to the pilot's likelihood_extremes.npz (default matches "
             "RunConfig.out_dir in fm_likelihood_extremes_v2.py).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("./e0a_outputs"))
    args = p.parse_args(argv)

    if not args.npz.exists():
        print(f"error: {args.npz} not found. Pass --npz /path/to/likelihood_extremes.npz",
              file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz)
    arrays = extract_replicates(data)
    decomp = variance_decomposition(arrays)
    pairs = pairwise_summary(arrays)
    rho_spread = correlation_spread(data, arrays)

    print("=" * 78)
    print("E0a: replicate-spread numerical noise floor")
    print("=" * 78)
    print(f"n fields               : {decomp['n_fields']}")
    print(f"n replicates per field : {decomp['n_replicates']}  ({', '.join(REPLICATE_KEYS)})")
    print(f"between-field SD       : {decomp['between_field_sd']:.4f} bits/dim "
          f"(pilot JSON reported {BETWEEN_FIELD_SD_REPORTED:.4f})")
    print(f"within-field SD        : {decomp['within_field_sd_numerical_noise']:.4f} bits/dim  <-- THE ANSWER")
    print(f"total pooled SD        : {decomp['total_sd_pooled']:.4f} bits/dim")
    print("-" * 78)
    for name, s in pairs.items():
        print(f"{name:28s} mean={s['mean_diff']:+.4f}  sd={s['sd_diff']:.4f}  "
              f"mean|diff|={s['mean_abs_diff']:.4f}  max|diff|={s['max_abs_diff']:.4f}")
    if rho_spread:
        print("-" * 78)
        print("Spearman rho, same covariate, 4 different noisy BPD realisations:")
        for cov_name, per_rep in rho_spread.items():
            vals = list(per_rep.values())
            print(f"  {cov_name:22s} " + "  ".join(f"{k}={v:.4f}" for k, v in per_rep.items())
                  + f"   (range {max(vals) - min(vals):.4f})")
    print("-" * 78)
    print(verdict(decomp["within_field_sd_numerical_noise"]))
    print("=" * 78)

    summary = {
        "n_fields": decomp["n_fields"],
        "n_replicates": decomp["n_replicates"],
        "between_field_sd": decomp["between_field_sd"],
        "within_field_sd_numerical_noise": decomp["within_field_sd_numerical_noise"],
        "total_sd_pooled": decomp["total_sd_pooled"],
        "pairwise": pairs,
        "rho_spread": rho_spread,
        "verdict": verdict(decomp["within_field_sd_numerical_noise"]),
    }
    with open(args.out_dir / "e0a_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    make_figure(decomp, args.out_dir / "e0a_replicate_spread.png")
    print(f"\nWrote {args.out_dir / 'e0a_summary.json'} "
          f"and {args.out_dir / 'e0a_replicate_spread.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
