#!/usr/bin/env python3
"""
table2_fdr_correction.py
===========================
Review concern #4 (major): Table 2 reports several tests of the same
hypothesis (the wet-fraction-independent relationship between BPD and field
extremity): the overall partial Spearman, the matched-pair Spearman and one
within-season partial Spearman per season (JF, MAM, JJAS, OND). None was
corrected for multiple comparisons.

WHAT THIS SCRIPT DOES
---------------------
It re-uses, unchanged, the bootstrap code that produced Table 2's own
confidence intervals, so p-values, CIs and q-values all come from one
procedure:

  * overall and within-season partial Spearman: the function
    bootstrap_within_season_cis() of scripts/make_figure2_what_density_measures.py.
    Design: the FULL 1461-day series is resampled with circular moving
    blocks (block length = 3 x the lag at which |ACF| of the BPD series first
    drops below 1/e, floor 15 d; 189 d on the real data), and every statistic
    (overall, and each season computed on the resampled rows of that
    season) is evaluated on the same resample.
  * matched-pair Spearman (optional, --matched-pairs): match_pairs() and
    moving_block_resample_indices() of scripts/analyze_e1_matched_pairs.py,
    re-matching inside every resample.

Table 2's published CIs came from these two scripts at their default of
300 resamples. Percentile CIs from 300 resamples carry Monte Carlo error of
about 0.01 to 0.02 at the endpoints, which matters for tests sitting near
zero. This script defaults to 10,000 resamples.

For each test a two-sided bootstrap p-value (2 x the smaller tail fraction of
the resampled estimates about zero, floored at 1/n_boot) is computed from the
same draws as the CI, then Benjamini-Hochberg (1995) q-values are computed
over a family of tests:

  family A: the four within-season tests only (m = 4)
  family B: every test in the table that was run (m = 5, or 6 with
            --matched-pairs). Family B is the one reported in the manuscript.

REQUIRES: e1_analysis_table.csv, and make_figure2_what_density_measures.py
(plus analyze_e1_matched_pairs.py for --matched-pairs) in --scripts-dir
(default: the folder this file sits in, i.e. scripts/). CPU only.

USAGE (from the project root)
-----------------------------
    python scripts/table2_fdr_correction.py --table e1_analysis_table.csv \
        --n-boot 10000 --matched-pairs --workers 4
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_common import bh_fdr, bootstrap_two_sided_pvalue  # noqa: E402


def _load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; pass --scripts-dir")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _matched_pair_chunk(args):
    """One worker: n_chunk re-matched block-bootstrap resamples."""
    scripts_dir, table, y_col, caliper, seed, n_chunk, tau = args
    mp = _load_module("mp_worker", Path(scripts_dir) / "analyze_e1_matched_pairs.py")
    df = pd.read_csv(table).sort_values("field_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n = len(df)
    out = []
    for _ in range(n_chunk):
        idx = mp.moving_block_resample_indices(n, tau, rng)
        s = mp.matched_pair_statistic(df.iloc[idx].reset_index(drop=True), caliper, y_col)
        if not np.isnan(s):
            out.append(s)
    return out


def _row(name: str, point: float, boot) -> dict:
    boot = np.asarray(boot, dtype=np.float64)
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"test": name, "n_valid_resamples": len(boot), "estimate": point,
            "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": bootstrap_two_sided_pvalue(boot, null_value=0.0)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=Path, default=Path("e1_analysis_table.csv"))
    p.add_argument("--y-col", type=str, default="bits_per_interior_dim")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--matched-pairs", action="store_true",
                   help="also bootstrap the matched-pair Spearman and include it in family B")
    p.add_argument("--caliper", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=1,
                   help="processes for the matched-pair bootstrap (the slow part)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=None,
                   help="output CSV (default: table2_fdr_corrected.csv next to --table)")
    args = p.parse_args()

    if not args.table.exists():
        print(f"error: {args.table} not found")
        return 1

    mf = _load_module("make_figure2", args.scripts_dir / "make_figure2_what_density_measures.py")
    df = pd.read_csv(args.table).sort_values("field_id").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {args.table}; n_boot={args.n_boot}, seed={args.seed}")

    print("Bootstrapping overall + within-season partial Spearman "
          "(full-series circular blocks, same code as Table 2)...")
    res, tau = mf.bootstrap_within_season_cis(df, args.y_col, args.n_boot,
                                              np.random.default_rng(args.seed))
    print(f"Block length: {tau} d\n")

    rows = [_row("overall", res["overall"]["point"], res["overall"]["boot"])]
    for s in ("JF", "MAM", "JJAS", "OND"):
        rows.append(_row(s, res[s]["point"], res[s]["boot"]))

    if args.matched_pairs:
        mp = _load_module("mp_main", args.scripts_dir / "analyze_e1_matched_pairs.py")
        pairs = mp.match_pairs(df, args.caliper, args.y_col)
        point = mp._spearman(pairs["delta_ext"].to_numpy(), pairs["delta_bpd"].to_numpy())
        tau_mp = mp.decorrelation_time(df[args.y_col].to_numpy(dtype=np.float64), min_days=15)
        print(f"Matched pairs: {len(pairs)} pairs, point estimate {point:.4f}, "
              f"block length {tau_mp} d; bootstrapping with re-matching "
              f"({args.workers} worker(s))...")
        per = int(np.ceil(args.n_boot / args.workers))
        jobs = [(str(args.scripts_dir), str(args.table), args.y_col, args.caliper,
                 args.seed + 1000 + i, per, tau_mp) for i in range(args.workers)]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as ex:
                parts = list(ex.map(_matched_pair_chunk, jobs))
        else:
            parts = [_matched_pair_chunk(jobs[0])]
        rows.append(_row("matched_pair", point, np.concatenate(parts)))

    result = pd.DataFrame(rows)

    season_mask = result["test"].isin(["JF", "MAM", "JJAS", "OND"])
    result["q_family_B"] = bh_fdr(result["p_value"].to_numpy())
    result["q_family_A"] = np.nan
    result.loc[season_mask, "q_family_A"] = bh_fdr(result.loc[season_mask, "p_value"].to_numpy())
    floor = 1.0 / args.n_boot

    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    pd.set_option("display.width", 200)
    print("=" * 100)
    print("Estimates, percentile 95% CIs, two-sided bootstrap p, BH q-values")
    print("=" * 100)
    print(result.to_string(index=False))
    print(f"\n(p-values are floored at 1/n_boot = {floor:.4f}; a p at the floor means no "
          f"resample fell on the other side of zero, so q there is a bound, not a value.)")
    m_b = len(result)
    n_b = int((result["q_family_B"] < args.alpha).sum())
    n_a = int((result.loc[season_mask, "q_family_A"] < args.alpha).sum())
    print(f"\nFamily A (4 seasons): {n_a}/4 survive q < {args.alpha}")
    print(f"Family B ({m_b} tests): {n_b}/{m_b} survive q < {args.alpha}")

    out_path = args.out if args.out is not None else args.table.parent / "table2_fdr_corrected.csv"
    result.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
