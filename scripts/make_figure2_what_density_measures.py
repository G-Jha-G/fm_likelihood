#!/usr/bin/env python3
"""
make_figure2_what_density_measures.py
========================================

Produces the brief's Figure 2 (Section 2, "What the density measures"),
exactly as specified: "(a) BPD vs wet fraction colored by season;
(b) BPD vs 99.9th percentile with the wet fraction partial removed;
(c) variance partition bars; (d) within season correlations with CI."

All four panels draw on E1's completed analysis (all four elements,
hierarchical R^2, partial correlation, within-season breakdown, matched
pairs). This script's only new computation is within-season CIs for
panel (d), which nothing built so far has produced, everything else
reuses already-validated logic from analyze_e1_confound_decomposition.py
and e1_block_bootstrap.py.

PANEL NOTES
-----------
(a) Raw scatter, wet_fraction vs BPD, colored by season. No adjustment,
    this panel exists specifically to show HOW dominant wet fraction is
    before any correction, the visual counterpart to the 98.4% R^2.

(b) Added-variable plot: BPD residualised on wet_fraction (rank-based,
    matching the partial correlation's own method) on the y-axis,
    extremity residualised on wet_fraction on the x-axis. The slope of
    the fitted line IS the partial relationship. Points colored by
    season for visual continuity with (a) and (d), even though season
    is NOT one of the controls in this specific panel (the brief's own
    spec says "wet fraction partial removed", singular; season is
    controlled separately, in the numeric partial correlation and in
    panel (d), not baked into this panel's residualisation).

(c) Incremental R^2 for each of the 4 covariate groups, in the brief's
    specified order.

(d) Forest plot: within-season partial Spearman (controlling for
    wet_fraction, computed separately per season, same design already
    validated in analyze_e1_confound_decomposition.py's
    within_season_spearman), each with a block-bootstrap 95% CI computed
    here for the first time. Overall estimate shown as a reference line.

USAGE
-----
    python scripts/make_figure2_what_density_measures.py
    python scripts/make_figure2_what_density_measures.py --n-boot 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coefs
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coefs


def _design_matrix(df: pd.DataFrame, columns: list[str], season_dummies: bool = False
                    ) -> np.ndarray:
    parts = [np.ones((len(df), 1))]
    for c in columns:
        if c == "season" and season_dummies:
            dummies = pd.get_dummies(df["season"], drop_first=True)
            parts.append(dummies.to_numpy(dtype=np.float64))
        elif c == "season":
            continue
        else:
            parts.append(df[[c]].to_numpy(dtype=np.float64))
    return np.hstack(parts)


GROUP1 = ["dry_fraction", "wet_fraction"]
GROUP2 = ["domain_mean", "wet_day_mean", "p999_wet", "spatial_max"]
GROUP3 = ["season", "day_of_year"]
GROUP4 = ["raps_log_slope", "n_connected_wet_components", "mean_component_size",
          "intensity_concentration_top1pct", "morans_i"]
EXTREMITY_COL = "p999_wet"
SEASON_ORDER = ["JF", "MAM", "JJAS", "OND"]
SEASON_COLORS = {"JF": "#4C72B0", "MAM": "#55A868", "JJAS": "#C44E52", "OND": "#8172B2"}


def sample_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    denom = float((x ** 2).sum())
    if denom == 0:
        return np.zeros(max_lag + 1)
    acf = np.empty(max_lag + 1)
    n = len(x)
    for k in range(max_lag + 1):
        acf[k] = float((x[: n - k] * x[k:]).sum()) / denom
    return acf


def decorrelation_time(x: np.ndarray, min_days: int = 15, max_lag: int = 200,
                        block_multiplier: float = 3.0) -> int:
    """Same validated 3x correction used throughout E1's bootstrap work."""
    max_lag = min(max_lag, len(x) - 1)
    acf = sample_acf(x, max_lag)
    threshold = 1.0 / np.e
    below = np.where(np.abs(acf[1:]) < threshold)[0]
    tau = int(below[0]) + 1 if len(below) > 0 else max_lag
    return max(int(round(tau * block_multiplier)), min_days)


def moving_block_resample_indices(n: int, block_length: int, rng: np.random.Generator
                                   ) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block_length)) % n for s in starts])
    return idx[:n]


def hierarchical_r2(df: pd.DataFrame, y_col: str) -> list[dict]:
    y = df[y_col].to_numpy(dtype=np.float64)
    cumulative_cols: list[str] = []
    rows = []
    prev_r2 = 0.0
    for group_name, group_cols in [("wet\nfraction", GROUP1), ("domain mean\nintensity", GROUP2),
                                    ("season", GROUP3), ("spatial\norganization", GROUP4)]:
        cumulative_cols += group_cols
        X = _design_matrix(df, cumulative_cols, season_dummies=True)
        r2 = ols_r2(y, X)
        rows.append({"group": group_name, "incremental_R2": r2 - prev_r2})
        prev_r2 = r2
    return rows


def within_season_partial_rho(sub: pd.DataFrame, y_col: str) -> float:
    if len(sub) < 10:
        return float("nan")
    y = sub[y_col].to_numpy(dtype=np.float64)
    ext = sub[EXTREMITY_COL].to_numpy(dtype=np.float64)
    wf = sub["wet_fraction"].to_numpy(dtype=np.float64)
    ry, rext, rwf = _rankdata(y), _rankdata(ext), _rankdata(wf)
    X = np.column_stack([np.ones(len(sub)), rwf])
    y_resid = ols_residuals(ry, X)
    ext_resid = ols_residuals(rext, X)
    denom = float(np.sqrt((y_resid ** 2).sum() * (ext_resid ** 2).sum()))
    return float((y_resid * ext_resid).sum() / denom) if denom > 0 else float("nan")


def overall_partial_rho(df: pd.DataFrame, y_col: str) -> float:
    y = df[y_col].to_numpy(dtype=np.float64)
    ext = df[EXTREMITY_COL].to_numpy(dtype=np.float64)
    ry, rext = _rankdata(y), _rankdata(ext)
    control_X = _design_matrix(df, ["wet_fraction", "season"], season_dummies=True)
    y_resid = ols_residuals(ry, control_X)
    ext_resid = ols_residuals(rext, control_X)
    denom = float(np.sqrt((y_resid ** 2).sum() * (ext_resid ** 2).sum()))
    return float((y_resid * ext_resid).sum() / denom) if denom > 0 else float("nan")


def bootstrap_within_season_cis(df: pd.DataFrame, y_col: str, n_boot: int,
                                 rng: np.random.Generator) -> dict:
    y = df[y_col].to_numpy(dtype=np.float64)
    tau = decorrelation_time(y, min_days=15)
    n = len(df)

    results = {s: {"point": within_season_partial_rho(df[df["season"] == s], y_col),
                    "boot": []} for s in SEASON_ORDER}
    results["overall"] = {"point": overall_partial_rho(df, y_col), "boot": []}

    for _ in range(n_boot):
        idx = moving_block_resample_indices(n, tau, rng)
        df_b = df.iloc[idx].reset_index(drop=True)
        for s in SEASON_ORDER:
            sub_b = df_b[df_b["season"] == s]
            val = within_season_partial_rho(sub_b, y_col)
            if not np.isnan(val):
                results[s]["boot"].append(val)
        val = overall_partial_rho(df_b, y_col)
        if not np.isnan(val):
            results["overall"]["boot"].append(val)

    for k in results:
        boot = np.array(results[k]["boot"])
        if len(boot) >= 10:
            results[k]["ci"] = tuple(np.percentile(boot, [2.5, 97.5]))
        else:
            results[k]["ci"] = (float("nan"), float("nan"))
    return results, tau


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=Path, default=Path("e1_analysis_table.csv"))
    p.add_argument("--y-col", type=str, default="bits_per_interior_dim")
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("figure2_what_density_measures"))
    args = p.parse_args()

    if not args.table.exists():
        print(f"error: {args.table} not found")
        return 1

    df = pd.read_csv(args.table).sort_values("field_id").reset_index(drop=True)
    y_col = args.y_col
    print(f"Loaded {len(df)} rows\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # --- Panel (a): raw BPD vs wet fraction, colored by season ---
    ax = axes[0, 0]
    for s in SEASON_ORDER:
        sub = df[df["season"] == s]
        ax.scatter(sub["wet_fraction"], sub[y_col], s=8, alpha=0.5,
                   color=SEASON_COLORS[s], label=s, edgecolor="none")
    ax.set_xlabel("wet fraction")
    ax.set_ylabel("bits per interior dimension")
    ax.set_title("(a) BPD vs wet fraction, by season")
    ax.legend(frameon=False, fontsize=8, markerscale=2)

    # --- Panel (b): added-variable plot, wet fraction removed ---
    ax = axes[0, 1]
    y = df[y_col].to_numpy(dtype=np.float64)
    ext = df[EXTREMITY_COL].to_numpy(dtype=np.float64)
    wf = df["wet_fraction"].to_numpy(dtype=np.float64)
    ry, rext, rwf = _rankdata(y), _rankdata(ext), _rankdata(wf)
    X_wf_only = np.column_stack([np.ones(len(df)), rwf])
    y_resid = ols_residuals(ry, X_wf_only)
    ext_resid = ols_residuals(rext, X_wf_only)
    for s in SEASON_ORDER:
        mask = (df["season"] == s).to_numpy()
        ax.scatter(ext_resid[mask], y_resid[mask], s=8, alpha=0.5,
                   color=SEASON_COLORS[s], edgecolor="none")
    coefs = np.linalg.lstsq(np.column_stack([np.ones(len(df)), ext_resid]), y_resid,
                             rcond=None)[0]
    xs = np.linspace(ext_resid.min(), ext_resid.max(), 50)
    ax.plot(xs, coefs[0] + coefs[1] * xs, color="black", lw=1.5)
    rho_wf_only = _spearman(ext_resid, y_resid)
    ax.set_xlabel("extremity, rank-residual after removing wet fraction")
    ax.set_ylabel("BPD, rank-residual after removing wet fraction")
    ax.set_title(f"(b) BPD vs extremity, wet fraction removed\n"
                 f"(rho={rho_wf_only:.3f}, wet fraction only, no season control here)")

    # --- Panel (c): variance partition bars ---
    ax = axes[1, 0]
    hvp = hierarchical_r2(df, y_col)
    groups = [r["group"] for r in hvp]
    vals = [r["incremental_R2"] for r in hvp]
    colors_c = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    ax.bar(groups, vals, color=colors_c)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("incremental R^2")
    ax.set_title("(c) Variance partition, added in this order")

    # --- Panel (d): within-season forest plot ---
    ax = axes[1, 1]
    print("Bootstrapping within-season CIs (this is the new computation)...")
    rng = np.random.default_rng(args.seed)
    results, tau = bootstrap_within_season_cis(df, y_col, args.n_boot, rng)
    print(f"Block length used: {tau} days\n")

    rows_order = ["overall"] + SEASON_ORDER
    y_positions = np.arange(len(rows_order))[::-1]
    for ypos, key in zip(y_positions, rows_order):
        point = results[key]["point"]
        lo, hi = results[key]["ci"]
        color = "black" if key == "overall" else SEASON_COLORS[key]
        ax.errorbar(point, ypos, xerr=[[point - lo], [hi - point]], fmt="o",
                    color=color, capsize=3, markersize=6)
        print(f"  {key:>8}: rho={point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([k.upper() if k != "overall" else "OVERALL" for k in rows_order])
    ax.set_xlabel("partial Spearman rho (extremity vs BPD | wet fraction, season)")
    ax.set_title("(d) Within-season partial correlation, with 95% CI")

    fig.suptitle("What the conditional density measures", fontsize=13, y=1.00)
    fig.tight_layout()
    for ext_fmt in ("png", "pdf"):
        out_path = args.out.with_suffix(f".{ext_fmt}")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
