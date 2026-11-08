#!/usr/bin/env python3
"""
make_figure3_conditioning_sensitivity.py
===========================================

Figure 3: the visual counterpart to E3's four-axis conditioning
sensitivity result. (a) magnitude, (b) shift, (c) blur: dose-response
curves, one line per field, BPD expressed RELATIVE TO EACH FIELD'S OWN
BASELINE, not absolute BPD. Fields range from very dry (BPD ~0.6) to
very wet (BPD ~3+); plotting absolute values would make the driest
fields' real sensitivity look artificially flat purely from y-axis
scale, exactly the kind of thing this project has been careful to avoid
throughout (Section 4.2's own analysis is built on relative, not
absolute, comparisons for the same reason). (d) swap: paired scatter of
baseline vs swap-conditioned BPD, not a dose curve, points above the
y=x line show BPD increased when a real different day's conditioning
was substituted.

Magnitude's x-axis uses SIGNED log2(level), deliberately, even though
the actual statistical test (analyze_e3_conditioning_sensitivity.py)
uses the FOLDED |log2(level)| distance. Showing the signed axis is what
makes the U-shape visible (and shows that it is steeper on the too-dry
side), which is the whole reason the folded distance was necessary in
the first place; a reader should be able to see why, not just be told.

Fields are colored by season throughout, matching Figure 2's
convention, both for visual consistency across the paper and because it
lets a reader check by eye whether conditioning sensitivity itself
varies by season, a pattern the current statistics don't test for.

USAGE
-----
    python scripts/make_figure3_conditioning_sensitivity.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEASON_COLORS = {"JF": "#4C72B0", "MAM": "#55A868", "JJAS": "#C44E52", "OND": "#8172B2"}
MAGNITUDE_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
SHIFT_LEVELS = [0, 1, 2, 4, 8, 16, 32]
BLUR_LEVELS = [0, 1, 2, 4, 8]


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def field_mean_by_level(rows: list[dict], axis: str, baseline_by_field: dict) -> dict:
    """field_id -> {level: mean_bpd_across_replicates}, including level
    corresponding to baseline (1.0 for magnitude, 0 for shift/blur)."""
    by_field_level = defaultdict(list)
    for r in rows:
        if r["axis"] == axis:
            by_field_level[(r["field_id"], r["level"])].append(r["bits_per_interior_dim"])

    out = defaultdict(dict)
    for (field_id, level), vals in by_field_level.items():
        out[field_id][level] = float(np.mean(vals))
    baseline_level = 1.0 if axis == "magnitude" else 0
    for field_id, bpd in baseline_by_field.items():
        out[field_id][baseline_level] = bpd
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("e3_outputs/e3_results_FINAL_corrected.jsonl"))
    p.add_argument("--fields-csv", type=Path, default=Path("e3_selected_fields.csv"))
    p.add_argument("--out", type=Path, default=Path("figure3_conditioning_sensitivity"))
    args = p.parse_args()

    if not args.results.exists() or not args.fields_csv.exists():
        print(f"error: need both {args.results} and {args.fields_csv}")
        return 1

    rows = load_results(args.results)
    fields_df = pd.read_csv(args.fields_csv).set_index("field_id")
    print(f"Loaded {len(rows)} results, {len(fields_df)} field metadata rows")

    baseline_by_field = defaultdict(list)
    for r in rows:
        if r["axis"] == "baseline":
            baseline_by_field[r["field_id"]].append(r["bits_per_interior_dim"])
    baseline_by_field = {f: float(np.mean(v)) for f, v in baseline_by_field.items()}

    mag = field_mean_by_level(rows, "magnitude", baseline_by_field)
    shift = field_mean_by_level(rows, "shift", baseline_by_field)
    blur = field_mean_by_level(rows, "blur", baseline_by_field)

    swap_by_field = defaultdict(list)
    for r in rows:
        if r["axis"] == "swap":
            swap_by_field[r["field_id"]].append(r["bits_per_interior_dim"])
    swap_by_field = {f: float(np.mean(v)) for f, v in swap_by_field.items()}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # --- Panel (a): magnitude, SIGNED log2 x-axis, relative BPD ---
    ax = axes[0, 0]
    for field_id, level_vals in mag.items():
        season = fields_df.loc[field_id, "season"] if field_id in fields_df.index else "JF"
        color = SEASON_COLORS.get(season, "grey")
        levels = sorted(level_vals.keys())
        xs = [np.log2(lv) for lv in levels]
        base = level_vals.get(1.0, level_vals[levels[len(levels)//2]])
        ys = [level_vals[lv] - base for lv in levels]
        ax.plot(xs, ys, color=color, alpha=0.5, lw=1.2)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("log2(magnitude factor)  [0 = truth]")
    ax.set_ylabel("BPD relative to own baseline")
    ax.set_title("(a) Magnitude: U-shaped, steeper when too dry\n(statistic uses the folded distance |log2 factor|)")

    # --- Panel (b): shift ---
    ax = axes[0, 1]
    for field_id, level_vals in shift.items():
        season = fields_df.loc[field_id, "season"] if field_id in fields_df.index else "JF"
        color = SEASON_COLORS.get(season, "grey")
        levels = sorted(level_vals.keys())
        base = level_vals.get(0, level_vals[levels[0]])
        ys = [level_vals[lv] - base for lv in levels]
        ax.plot(levels, ys, color=color, alpha=0.5, lw=1.2)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("shift (pixels)")
    ax.set_ylabel("BPD relative to own baseline")
    ax.set_title("(b) Shift: monotonic response")

    # --- Panel (c): blur ---
    ax = axes[1, 0]
    for field_id, level_vals in blur.items():
        season = fields_df.loc[field_id, "season"] if field_id in fields_df.index else "JF"
        color = SEASON_COLORS.get(season, "grey")
        levels = sorted(level_vals.keys())
        base = level_vals.get(0, level_vals[levels[0]])
        ys = [level_vals[lv] - base for lv in levels]
        ax.plot(levels, ys, color=color, alpha=0.5, lw=1.2)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("blur sigma (pixels)")
    ax.set_ylabel("BPD relative to own baseline")
    ax.set_title("(c) Blur: dip, then rise\n(mild blur decreases BPD, aggressive blur increases it)")

    # --- Panel (d): swap, paired scatter ---
    ax = axes[1, 1]
    for field_id, swap_bpd in swap_by_field.items():
        season = fields_df.loc[field_id, "season"] if field_id in fields_df.index else "JF"
        color = SEASON_COLORS.get(season, "grey")
        ax.scatter(baseline_by_field[field_id], swap_bpd, color=color, s=30,
                   edgecolor="none", alpha=0.85, zorder=3)
    all_vals = list(baseline_by_field.values()) + list(swap_by_field.values())
    lims = [min(all_vals) * 0.9, max(all_vals) * 1.05]
    ax.plot(lims, lims, color="grey", lw=1, ls="--", zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("baseline BPD (true conditioning)")
    ax.set_ylabel("swap BPD (different day's conditioning)")
    ax.set_title("(d) Swap: all 20 fields above the line")

    handles = [plt.Line2D([0], [0], color=c, lw=2, label=s) for s, c in SEASON_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Conditioning sensitivity across four perturbation axes", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    for ext in ("png", "pdf"):
        out_path = args.out.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
