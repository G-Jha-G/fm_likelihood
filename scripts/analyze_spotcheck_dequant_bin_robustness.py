#!/usr/bin/env python3
"""
analyze_spotcheck_dequant_bin_robustness.py
==============================================

Reads spotcheck_e0bc_dequant_bin_robustness.py's output and directly
answers: do E0b's SD-across-seeds and E0c's bias-vs-reference come out
the same under both the old (0.1) and corrected (0.0625) dequant bin?

If yes: the "common random numbers make the log-Jacobian term cancel"
argument holds, and E0b/E0c's original conclusions (K=1 sufficient,
rk4-100 biased by ~0.1, dopri5 corrects it) stand as originally
reported, no rerun needed.

If no: the argument fails here the same way it failed for E1's
"constant shift" claim, and E0b/E0c need the same full-recompute
treatment E1 already got.

USAGE
-----
    python scripts/analyze_spotcheck_dequant_bin_robustness.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_PATH = Path("spotcheck_outputs/spotcheck_results.jsonl")
FINEST = "dopri5_rtol1e-6"


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_e0b(rows: list[dict]) -> None:
    print("=" * 88)
    print("E0B-STYLE CHECK: does the per-field SD across seeds (K=1, rk4-100) match")
    print("under the old and corrected dequant bin?")
    print("=" * 88)
    by_field_bin = defaultdict(list)
    for r in rows:
        if r["check"] != "e0b":
            continue
        by_field_bin[(r["field_id"], r["bin_label"])].append(r["bits_per_interior_dim"])

    fields = sorted({f for f, _ in by_field_bin})
    print(f"{'field':>6} {'bin':>5} {'n':>3} {'mean_bpd':>10} {'SD':>8}")
    sds = {}
    for field in fields:
        for bin_label in ("old", "new"):
            vals = by_field_bin.get((field, bin_label), [])
            if not vals:
                continue
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            sds[(field, bin_label)] = sd
            print(f"{field:>6} {bin_label:>5} {len(vals):>3} {np.mean(vals):>10.4f} {sd:>8.5f}")

    print("\nComparison (does K=1 stay sufficient under the corrected bin?):")
    for field in fields:
        sd_old = sds.get((field, "old"))
        sd_new = sds.get((field, "new"))
        if sd_old is None or sd_new is None:
            continue
        verdict = "CONSISTENT" if abs(sd_new - sd_old) < 0.01 else "DIFFERS, investigate"
        print(f"  field {field}: SD_old={sd_old:.5f}  SD_new={sd_new:.5f}  "
              f"diff={sd_new - sd_old:+.5f}  [{verdict}]")


def analyze_e0c(rows: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("E0C-STYLE CHECK: does the bias vs the dopri5-1e-6 reference match")
    print("under the old and corrected dequant bin?")
    print("=" * 88)
    value = {}
    for r in rows:
        if r["check"] != "e0c":
            continue
        key = (r["field_id"], r["replicate"], r["solver_name"], r["bin_label"])
        value[key] = r["bits_per_interior_dim"]

    fields = sorted({f for f, _, _, _ in value})
    reps = sorted({rp for _, rp, _, _ in value})
    solvers = sorted({s for _, _, s, _ in value if s != FINEST})

    print(f"{'field':>6} {'solver':>18} {'bin':>5} {'mean_bias':>10}")
    biases = {}
    for field in fields:
        for solver in solvers:
            for bin_label in ("old", "new"):
                diffs = []
                for rep in reps:
                    k = (field, rep, solver, bin_label)
                    ref_k = (field, rep, FINEST, bin_label)
                    if k in value and ref_k in value:
                        diffs.append(value[k] - value[ref_k])
                if not diffs:
                    continue
                mean_bias = float(np.mean(diffs))
                biases[(field, solver, bin_label)] = mean_bias
                print(f"{field:>6} {solver:>18} {bin_label:>5} {mean_bias:>10.4f}")

    print("\nComparison (does the bias magnitude and direction survive the bin correction?):")
    for field in fields:
        for solver in solvers:
            b_old = biases.get((field, solver, "old"))
            b_new = biases.get((field, solver, "new"))
            if b_old is None or b_new is None:
                continue
            verdict = "CONSISTENT" if abs(b_new - b_old) < 0.02 else "DIFFERS, investigate"
            print(f"  field {field}, {solver}: bias_old={b_old:+.4f}  bias_new={b_new:+.4f}  "
                  f"diff={b_new - b_old:+.4f}  [{verdict}]")


def main():
    if not RESULTS_PATH.exists():
        print(f"error: {RESULTS_PATH} not found")
        return 1
    rows = load_results(RESULTS_PATH)
    print(f"Loaded {len(rows)} results\n")
    analyze_e0b(rows)
    analyze_e0c(rows)
    print("\n" + "=" * 88)
    print("If everything above says CONSISTENT: E0b and E0c's original conclusions stand,")
    print("no rerun needed, only E1 required the full recompute it already got.")
    print("If anything says DIFFERS: that specific claim needs the same treatment E1 got.")
    print("=" * 88)


if __name__ == "__main__":
    raise SystemExit(main())
