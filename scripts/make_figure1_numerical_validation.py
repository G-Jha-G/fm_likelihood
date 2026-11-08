#!/usr/bin/env python3
"""
make_figure1_numerical_validation.py
=======================================

Figure 1 for the standalone GMD paper: the numerical-validation result
that Methods currently only asserts. E0c measured, via common-random-
number pairing against a converged reference (dopri5 at rtol=1e-6),
the discretization error of every tested solver setting across 6 fields
(3 wet-fraction extremes, 3 intensity extremes) and 3 replicates each.
This turns that already-computed data into the figure a reviewer will
expect: bias against the finest setting (panel a) and computational
cost (panel b), together making the paper's central methods claim
visible rather than just asserted, dopri5 is both more accurate and
cheaper than any tested fixed-step RK4 setting.

USAGE
-----
    python scripts/make_figure1_numerical_validation.py
    python scripts/make_figure1_numerical_validation.py --results e0c_outputs/e0c_results.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINEST = "dopri5_rtol1e-06"
TOLERANCE = 0.02

SOLVER_ORDER = ["rk4_n10", "rk4_n25", "rk4_n50", "rk4_n100", "rk4_n200",
                "dopri5_rtol1e-04", "dopri5_rtol1e-05", "dopri5_rtol1e-06"]
SOLVER_LABELS = {"rk4_n10": "RK4\nn=10", "rk4_n25": "RK4\nn=25", "rk4_n50": "RK4\nn=50",
                  "rk4_n100": "RK4\nn=100", "rk4_n200": "RK4\nn=200",
                  "dopri5_rtol1e-04": "dopri5\nrtol=1e-4", "dopri5_rtol1e-05": "dopri5\nrtol=1e-5",
                  "dopri5_rtol1e-06": "dopri5\nrtol=1e-6\n(reference)"}
IS_RK4 = {s: s.startswith("rk4") for s in SOLVER_ORDER}


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
    p.add_argument("--results", type=Path, default=Path("e0c_outputs/e0c_results.jsonl"))
    p.add_argument("--out", type=Path, default=Path("figure1_numerical_validation"))
    args = p.parse_args()

    if not args.results.exists():
        print(f"error: {args.results} not found")
        return 1

    rows = load_results(args.results)
    print(f"Loaded {len(rows)} results from {args.results}")

    value = {}
    nfe = defaultdict(list)
    for r in rows:
        value[(r["field_id"], r["replicate"], r["solver_name"])] = r["bits_per_interior_dim"]
        nfe[r["solver_name"]].append(r["n_function_evals"])

    fields = sorted({fid for fid, _, _ in value})
    replicates = sorted({rep for _, rep, _ in value})

    mean_abs_err, sem_abs_err = {}, {}
    for solver in SOLVER_ORDER:
        if solver == FINEST:
            mean_abs_err[solver] = 0.0
            sem_abs_err[solver] = 0.0
            continue
        errs = []
        for field in fields:
            for rep in replicates:
                ref_key = (field, rep, FINEST)
                k = (field, rep, solver)
                if ref_key in value and k in value:
                    errs.append(value[k] - value[ref_key])
        errs = np.array(errs)
        mean_abs_err[solver] = float(np.abs(errs).mean())
        sem_abs_err[solver] = float(np.abs(errs).std(ddof=1) / np.sqrt(len(errs)))
        print(f"  {solver:>20}: mean|bias|={mean_abs_err[solver]:.4f}  "
              f"n_pairs={len(errs)}  mean_nfe={np.mean(nfe[solver]):.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    xs = np.arange(len(SOLVER_ORDER))
    means = [mean_abs_err[s] for s in SOLVER_ORDER]
    sems = [sem_abs_err[s] for s in SOLVER_ORDER]
    colors = ["#C44E52" if IS_RK4[s] else "#4C72B0" for s in SOLVER_ORDER]
    ax.bar(xs, means, yerr=sems, capsize=3, color=colors)
    ax.axhline(TOLERANCE, color="black", ls="--", lw=1, label=f"tolerance ({TOLERANCE})")
    ax.set_xticks(xs)
    ax.set_xticklabels([SOLVER_LABELS[s] for s in SOLVER_ORDER], fontsize=8)
    ax.set_ylabel("mean |bias| vs converged reference\n(bits per interior dimension)")
    ax.set_title("(a) Discretization error by solver setting")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    costs = [np.mean(nfe[s]) for s in SOLVER_ORDER]
    ax.bar(xs, costs, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels([SOLVER_LABELS[s] for s in SOLVER_ORDER], fontsize=8)
    ax.set_ylabel("network evaluations per field (nfe)")
    ax.set_title("(b) Computational cost by solver setting")

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#C44E52", label="fixed-step RK4"),
                    Patch(facecolor="#4C72B0", label="adaptive dopri5")]
    ax.legend(handles=legend_elems, frameon=False, fontsize=9)

    fig.suptitle("Numerical validation of the exact-likelihood estimator", fontsize=13)
    fig.tight_layout()
    for ext_fmt in ("png", "pdf"):
        out_path = args.out.with_suffix(f".{ext_fmt}")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
