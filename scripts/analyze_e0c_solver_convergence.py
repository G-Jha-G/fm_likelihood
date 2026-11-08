#!/usr/bin/env python3
"""
analyze_e0c_solver_convergence.py
====================================

For each field and replicate, subtracts the finest setting's result
(dopri5 at the tightest tested rtol) from every other setting's result,
using the common-random-number pairing so the difference reflects
discretization error, not resampling noise. Then, across replicates,
computes a mean paired difference and a t-distribution 95% CI on the true
bias for each (field, solver setting), and a pooled-across-fields version
for an overall read.

Cost is read directly from n_function_evals in the results file (not
assumed from theory), so the "cheapest setting within tolerance" pick is
based on what was actually measured on your hardware, matching how E0b's
real per-unit cost differed from the naive nfe-based projection.

USAGE
-----
    python scripts/analyze_e0c_solver_convergence.py
    python scripts/analyze_e0c_solver_convergence.py --tolerance 0.02
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

FINEST = "dopri5_rtol1e-06"


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def t_ci_on_mean(diffs: list[float], alpha: float = 0.05):
    """Standard t-based CI on the mean of paired differences. Returns
    (mean, ci_lower, ci_upper, df). (nan, nan, nan, 0) if n < 2."""
    n = len(diffs)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"), 0)
    if not _HAS_SCIPY:
        raise RuntimeError("scipy required; pip install scipy --break-system-packages")
    arr = np.asarray(diffs, dtype=np.float64)
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return (mean, mean - tcrit * sem, mean + tcrit * sem, df)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("e0c_outputs/e0c_results.jsonl"))
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="max |error vs finest| (upper CI bound) to call a setting acceptable; "
                        "default matches E0b's target for consistency, not independently "
                        "re-derived from the brief's Section 1b 'smallest claimed effect' rule, "
                        "adjust if that rule implies a different number once known")
    args = p.parse_args()

    if not args.results.exists():
        print(f"error: {args.results} not found")
        return 1
    if not _HAS_SCIPY:
        print("error: scipy required. pip install scipy --break-system-packages")
        return 1

    rows = load_results(args.results)
    print(f"Loaded {len(rows)} results from {args.results}\n")

    # value[(field, replicate, solver)] = bits_per_interior_dim
    value = {}
    nfe = defaultdict(list)   # solver -> [nfe, nfe, ...]
    cost = defaultdict(list)  # solver -> [elapsed_sec, ...]
    for r in rows:
        value[(r["field_id"], r["replicate"], r["solver_name"])] = r["bits_per_interior_dim"]
        nfe[r["solver_name"]].append(r["n_function_evals"])
        cost[r["solver_name"]].append(r["elapsed_sec"])

    fields = sorted({fid for fid, _, _ in value})
    replicates = sorted({rep for _, rep, _ in value})
    solvers = sorted({s for _, _, s in value})

    if FINEST not in solvers:
        print(f"error: reference setting '{FINEST}' not found in results, cannot compute "
              f"error vs finest. Solvers present: {solvers}")
        return 1

    # per-field, per-solver: paired differences across replicates
    diffs_by_field_solver = defaultdict(list)
    for field in fields:
        for rep in replicates:
            ref_key = (field, rep, FINEST)
            if ref_key not in value:
                continue
            ref_val = value[ref_key]
            for solver in solvers:
                if solver == FINEST:
                    continue
                k = (field, rep, solver)
                if k in value:
                    diffs_by_field_solver[(field, solver)].append(value[k] - ref_val)

    print("=" * 100)
    print(f"PER-FIELD ERROR vs FINEST ({FINEST}), t-based 95% CI (tolerance: {args.tolerance})")
    print("=" * 100)
    print(f"{'field':>6} {'solver':>18} {'n':>3} {'df':>3} {'mean_err':>10} "
          f"{'CI_lower':>10} {'CI_upper':>10}  verdict")
    print("-" * 100)
    passes_by_solver = defaultdict(set)
    for field in fields:
        for solver in solvers:
            if solver == FINEST:
                continue
            diffs = diffs_by_field_solver.get((field, solver), [])
            mean, lo, hi, df = t_ci_on_mean(diffs)
            if df == 0:
                verdict = f"n={len(diffs)}, insufficient for a CI"
            elif abs(lo) < args.tolerance and abs(hi) < args.tolerance:
                verdict = "PASS (CI entirely within tolerance)"
                passes_by_solver[solver].add(field)
            else:
                verdict = "NOT ESTABLISHED (CI extends past tolerance)"
            mean_str = f"{mean:+.5f}" if df > 0 else "  n/a  "
            lo_str = f"{lo:+.5f}" if df > 0 else "  n/a  "
            hi_str = f"{hi:+.5f}" if df > 0 else "  n/a  "
            print(f"{field:>6} {solver:>18} {len(diffs):>3} {df:>3} {mean_str:>10} "
                  f"{lo_str:>10} {hi_str:>10}  {verdict}")
    print()

    print("=" * 100)
    print("COST PER SOLVER SETTING (measured, not theoretical)")
    print("=" * 100)
    print(f"{'solver':>18} {'mean_nfe':>10} {'mean_sec':>10}")
    for solver in solvers:
        print(f"{solver:>18} {np.mean(nfe[solver]):>10.0f} {np.mean(cost[solver]):>10.1f}")
    print()

    print("=" * 100)
    print("RECOMMENDATION: cheapest setting passing on EVERY field tested")
    print("=" * 100)
    all_fields = set(fields)
    fully_passing = [s for s in solvers if s != FINEST and passes_by_solver.get(s) == all_fields]
    if fully_passing:
        # rank by nfe, not measured seconds: nfe is deterministic, wall-clock
        # timing has measurement noise that can dominate at small scale (this
        # was caught during testing, not assumed away). On real hardware at
        # the scale this project runs at, E0b showed cost tracks nfe almost
        # perfectly linearly anyway, so this rarely changes the answer, but
        # it removes a real failure mode rather than trusting noisy timing.
        cheapest = min(fully_passing, key=lambda s: np.mean(nfe[s]))
        print(f"Settings passing on all {len(all_fields)} fields: {fully_passing}")
        print(f"Cheapest of those by nfe (measured seconds shown for reference, not used "
              f"for ranking): {cheapest}, ~{np.mean(nfe[cheapest]):.0f} nfe/field, "
              f"~{np.mean(cost[cheapest]):.1f}s/field measured")
    else:
        partial = {s: sorted(f) for s, f in passes_by_solver.items()}
        print("No setting yet passes on every field tested. Per-solver partial pass fields:")
        for s, f in sorted(partial.items()):
            print(f"  {s}: passes on {f} of {sorted(all_fields)}")
        print("More replicates would narrow the CIs above; the cheap RK4 settings "
              "(n=10, n=25) are the ones most likely to still be borderline, add "
              "replicates there first rather than uniformly across all settings.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
