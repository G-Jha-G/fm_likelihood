#!/usr/bin/env python3
"""
make_figure4_structural_sensitivity.py
=========================================

Figure 4: E2's structural sensitivity result. Panels (a) and (b) show
raw per-field dose-response curves (BPD relative to each field's own
baseline), matching Figure 3's convention. Panel (c) exists specifically
because panel (a)'s blur curves show a valley shape that a spaghetti
plot alone doesn't convey statistical significance for: it shows the
five key phases (mild/moderate/plateau/strong/aggressive) as a
forest-style plot with field-bootstrap 95% CIs, the same statistics
reported in Table 4. Panel (d) shows the strongest cross-experiment
correlation found (wet fraction vs the aggressive-blur reversal),
flagged in the text as exploratory (n=20, no CI computed for this
specific point estimate), not confirmed with the same rigor as the two
primary axes.

USAGE
-----
    python scripts/make_figure4_structural_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEASON_COLORS = {"JF": "#4C72B0", "MAM": "#55A868", "JJAS": "#C44E52", "OND": "#8172B2"}
BLUR_PHASES = [("mild", 1), ("moderate", 4), ("plateau", 8), ("strong", 16), ("aggressive", 32)]


def field_bootstrap_ci(vals_dict: dict, n_boot: int, rng: np.random.Generator,
                        alpha: float = 0.05):
    vals = np.array([v for v in vals_dict.values() if not np.isnan(v)])
    n = len(vals)
    point = float(vals.mean())
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = vals[idx].mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), n


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
    p.add_argument("--results", type=Path, default=Path("e2_outputs/e2_results.jsonl"))
    p.add_argument("--fields-csv", type=Path, default=Path("e3_selected_fields.csv"),
                   help="E2 reuses E3's field selection, including season and wet_fraction")
    p.add_argument("--n-boot", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("figure4_structural_sensitivity"))
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

    def field_mean_by_level(axis: str) -> dict:
        by_field_level = defaultdict(list)
        for r in rows:
            if r["axis"] == axis:
                by_field_level[(r["field_id"], r["level"])].append(r["bits_per_interior_dim"])
        out = defaultdict(dict)
        for (fid, lvl), vals in by_field_level.items():
            out[fid][lvl] = float(np.mean(vals))
        for fid, bpd in baseline_by_field.items():
            out[fid][0] = bpd
        return out

    blur_by_field = field_mean_by_level("blur")
    shuffle_by_field = field_mean_by_level("shuffle")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    for fid, level_vals in blur_by_field.items():
        season = fields_df.loc[fid, "season"] if fid in fields_df.index else "JF"
        color = SEASON_COLORS.get(season, "grey")
        levels = sorted(level_vals.keys())
        base = level_vals[0]
        ys = [level_vals[lv] - base for lv in levels]
        ax.plot(levels, ys, color=color, alpha=0.5, lw=1.2)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("blur sigma (pixels)")
    ax.set_ylabel("BPD relative to own baseline")
    ax.set_title("(a) Target blur: valley then reversal\n(not monotonic, see panel c)")

    ax = axes[0, 1]
    for fid, level_vals in shuffle_by_field.items():
        season = fields_df.loc[fid, "season"] if fid in fields_df.index else "JF"
        color = SEASON_COLORS.get(season, "grey")
        levels = sorted(level_vals.keys())
        base = level_vals[0]
        ys = [level_vals[lv] - base for lv in levels]
        ax.plot(levels, ys, color=color, alpha=0.5, lw=1.2)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("shuffle block size (pixels)")
    ax.set_ylabel("BPD relative to own baseline")
    ax.set_title("(b) Block-shuffle: strong, near-universal response\n(0/60 negative at every level)")

    ax = axes[1, 0]
    rng = np.random.default_rng(args.seed)
    print("\nBootstrapping blur phase CIs...")
    y_positions = np.arange(len(BLUR_PHASES))[::-1]
    for ypos, (label, lvl) in zip(y_positions, BLUR_PHASES):
        per_field = {fid: level_vals[lvl] - level_vals[0]
                     for fid, level_vals in blur_by_field.items() if lvl in level_vals}
        point, lo, hi, n = field_bootstrap_ci(per_field, args.n_boot, rng)
        ax.errorbar(point, ypos, xerr=[[point - lo], [hi - point]], fmt="o",
                    color="#4C72B0" if point < 0 else "#C44E52", capsize=3, markersize=6)
        print(f"  {label:>10} (sigma={lvl}): mean={point:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{lbl}\n(sigma={lvl})" for lbl, lvl in BLUR_PHASES])
    ax.set_xlabel("mean BPD relative to baseline")
    ax.set_title("(c) Blur phases, with 95% CI\n(the valley shape, statistically)")

    ax = axes[1, 1]
    blur32_delta = {fid: level_vals[32] - level_vals[0]
                     for fid, level_vals in blur_by_field.items() if 32 in level_vals}
    for fid, delta in blur32_delta.items():
        if fid not in fields_df.index:
            continue
        season = fields_df.loc[fid, "season"]
        wf = fields_df.loc[fid, "wet_fraction"]
        ax.scatter(wf, delta, color=SEASON_COLORS.get(season, "grey"), s=30,
                   edgecolor="none", alpha=0.85, zorder=3)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("wet fraction (true field)")
    ax.set_ylabel("BPD delta at aggressive blur (sigma=32)")
    ax.set_title("(d) Wetter fields lose more to aggressive blur\n(exploratory, n=20, rho=0.61, no CI)")

    handles = [plt.Line2D([0], [0], color=c, lw=2, label=s) for s, c in SEASON_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Structural sensitivity of the target field", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    for ext in ("png", "pdf"):
        out_path = args.out.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
