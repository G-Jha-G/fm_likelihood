#!/usr/bin/env python3
"""
make_figure5_screening.py
=========================
Figure for the screening test: AUC (probability that a perturbed field scores
worse than a real day, after removing wet fraction and season) against
perturbation level, with field-bootstrap 95% intervals.

Reads the CSV written by screening_test.py and draws one panel per axis found:

  conditioning file (name contains "e3"): magnitude, shift, blur, and swap + control
  target file       (name contains "e2"): block-shuffle and target blur (only if present)

The unperturbed control (axis "baseline") is drawn in grey at the no-change
position of each panel, so the reader can see where "no separation" sits.

USAGE (from the project root; CPU only, seconds)
-----
    python scripts/make_figure5_screening.py --csv screening_test_results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#1f5f99"
GREY = "#7a7a7a"


def _panel(ax, sub, ctrl, order, labels, title, xlabel):
    """sub: rows for one axis with numeric 'level'; ctrl: control row or None."""
    xs, ys, lo, hi, cols = [], [], [], [], []
    pos = 0
    for lev, lab in zip(order, labels):
        if lev is None:                       # control position
            if ctrl is None:
                continue
            r = ctrl
            cols.append(GREY)
        else:
            m = sub[np.isclose(sub["level_num"], lev)]
            if m.empty:
                continue
            r = m.iloc[0]
            cols.append(BLUE)
        xs.append(pos)
        ys.append(r["AUC_high"])
        lo.append(r["AUC_high"] - r["AUC_high_lo"])
        hi.append(r["AUC_high_hi"] - r["AUC_high"])
        pos += 1
    used = [lab for lev, lab in zip(order, labels)
            if (lev is None and ctrl is not None) or
               (lev is not None and not sub[np.isclose(sub["level_num"], lev)].empty)]
    for x, y, l, h, c in zip(xs, ys, lo, hi, cols):
        ax.errorbar(x, y, yerr=[[l], [h]], fmt="o", color=c, ecolor=c, capsize=3,
                    markersize=5, lw=1.3)
    ax.plot(xs, ys, color=BLUE, lw=0.8, alpha=0.5, zorder=0)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(used, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, max(len(xs) - 0.5, 0.5))
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", lw=0.3, alpha=0.5)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("screening_test_results.csv"))
    p.add_argument("--out", type=Path, default=Path("figure5_screening"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["level_num"] = pd.to_numeric(df["level"], errors="coerce")
    cond = df[df["file"].str.contains("e3", case=False)]
    targ = df[df["file"].str.contains("e2", case=False)]

    panels = []   # (title, builder)
    ctrl_c = cond[cond["axis"] == "baseline"]
    ctrl_c = ctrl_c.iloc[0] if len(ctrl_c) else None
    ctrl_t = targ[targ["axis"] == "baseline"]
    ctrl_t = ctrl_t.iloc[0] if len(ctrl_t) else None

    def fmt(v):
        return f"{v:g}"

    mag = cond[cond["axis"] == "magnitude"]
    if len(mag):
        lv = [0.25, 0.5, 0.75, None, 1.5, 2.0, 4.0]
        panels.append(("(a) Conditioning magnitude", mag, ctrl_c, lv,
                       [("×" + fmt(v)) if v else "none" for v in lv], "scale factor"))
    sh = cond[cond["axis"] == "shift"]
    if len(sh):
        lv = [None, 1, 2, 4, 8, 16, 32]
        panels.append(("(b) Conditioning shift", sh, ctrl_c, lv,
                       [fmt(v) if v else "none" for v in lv], "shift (pixels)"))
    bl = cond[cond["axis"] == "blur"]
    if len(bl):
        lv = [None, 1, 2, 4, 8]
        panels.append(("(c) Conditioning blur", bl, ctrl_c, lv,
                       [fmt(v) if v else "none" for v in lv], "blur σ (pixels)"))
    sw = cond[cond["axis"] == "swap"]
    if len(sw):
        sw = sw.copy()
        sw["level_num"] = 1.0
        panels.append(("(d) Different day's conditioning", sw, ctrl_c, [None, 1.0],
                       ["none", "swap"], ""))
    shf = targ[targ["axis"] == "shuffle"]
    if len(shf):
        lv = [None, 2, 4, 8, 16, 32]
        panels.append(("(e) Target block-shuffle", shf, ctrl_t, lv,
                       [fmt(v) if v else "none" for v in lv], "block size (pixels)"))
    tb = targ[targ["axis"] == "blur"]
    if len(tb):
        lv = [None, 1, 2, 4, 8, 16, 32]
        panels.append(("(f) Target blur", tb, ctrl_t, lv,
                       [fmt(v) if v else "none" for v in lv], "blur σ (pixels)"))

    n = len(panels)
    ncol = 2 if n <= 4 else 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    for ax, (title, sub, ctrl, lv, lab, xl) in zip(axes.ravel(), panels):
        _panel(ax, sub, ctrl, lv, lab, title, xl)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for r in range(nrow):
        axes[r, 0].set_ylabel("AUC vs real days", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=220)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"Wrote {args.out.with_suffix('.png')} and .pdf ({n} panels)")


if __name__ == "__main__":
    main()
