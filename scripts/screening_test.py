#!/usr/bin/env python3
"""
screening_test.py
=================
Does the conditional likelihood work as a SCREEN? Can it separate real fields
(paired with their true conditioning) from degraded or mis-conditioned ones?

The paper shows the score responds to perturbations, but that is not the same
as showing it can flag them. This script turns the existing perturbation runs
into a detection test, with no new model runs.

DESIGN
------
Reference ("real") set R: every test day in e1_analysis_table.csv that is NOT
one of the perturbed fields. Each has a BPD and a wet fraction and season.

Because BPD is dominated by wet fraction (R^2 = 0.986), a raw threshold on BPD
would just detect wet days. So the screening score is the BPD RESIDUAL after
removing what wet fraction and season predict:

    score = BPD - f(wet_fraction, season),   f = cubic in wet fraction + season

f is fitted on R only. The residuals of R itself are computed out of fold
(5 contiguous time blocks, respecting autocorrelation) so the reference is not
optimistic. Perturbed samples are scored with f fitted on all of R, using the
wet fraction and season of the ORIGINAL field.

For every (file, axis, level) group of perturbed samples this reports:
  * mean score (bits per dimension) and in units of the reference SD
  * AUC_high  = P(score_perturbed > score_reference)   (0.5 = no separation)
  * AUC_abs   = same with |score|, i.e. direction-agnostic
  * TPR at 5% FPR (upper tail), the fraction flagged when only 5% of real days are
  * the UNPERTURBED baseline rows of the same fields as a control: AUC near 0.5

Uncertainty: percentile CI from a bootstrap over FIELDS (the sampling unit of the
perturbation experiments; all replicates of a field travel together). The
reference set is treated as fixed (1441 days, much larger than 20 fields).

CAVEATS PRINTED WITH THE RESULT
  * Conditioning perturbations (magnitude, shift, blur, swap) leave the scored
    target untouched, so its wet fraction is exact.
  * Block-shuffle conserves wet fraction exactly. Target blur changes it a little,
    and here the original wet fraction is used (approximate).
  * Only 20 fields: AUC intervals are wide. Read them, not just the point values.

USAGE (from the project root; CPU only, runs in seconds)
-----
    python scripts/screening_test.py --table e1_analysis_table.csv \
        --results e3_outputs/e3_results_FINAL_corrected.jsonl \
        --results e2_outputs/e2_results.jsonl

Pass every jsonl you want scored (the E3 conditioning file and the E2 target
structure file). Each needs columns field_id, axis, level, bits_per_interior_dim,
and optionally replicate. Rows with axis == "baseline" are used as the control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = ["JF", "MAM", "JJAS", "OND"]


def design(wf: np.ndarray, season: np.ndarray) -> np.ndarray:
    cols = [np.ones_like(wf), wf, wf ** 2, wf ** 3]
    for s in SEASONS[1:]:
        cols.append((season == s).astype(float))
    return np.column_stack(cols)


def fit(wf, season, y):
    beta, *_ = np.linalg.lstsq(design(wf, season), y, rcond=None)
    return beta


def predict(beta, wf, season):
    return design(wf, season) @ beta


def oof_residuals(df: pd.DataFrame, k: int = 5) -> np.ndarray:
    """Out-of-fold residuals using k contiguous time blocks."""
    df = df.sort_values("field_id").reset_index(drop=True)
    n = len(df)
    edges = np.linspace(0, n, k + 1).astype(int)
    wf = df["wet_fraction"].to_numpy(float)
    se = df["season"].to_numpy()
    y = df["bits_per_interior_dim"].to_numpy(float)
    res = np.empty(n)
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.arange(n)[a:b]
        train = np.setdiff1d(np.arange(n), test)
        beta = fit(wf[train], se[train], y[train])
        res[test] = y[test] - predict(beta, wf[test], se[test])
    return res


def auc_contrib(scores: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """Per-sample P(score > ref) with ties counted half."""
    lo = np.searchsorted(ref_sorted, scores, side="left")
    hi = np.searchsorted(ref_sorted, scores, side="right")
    return (lo + 0.5 * (hi - lo)) / len(ref_sorted)


def summarise(group: pd.DataFrame, ref_res: np.ndarray, ref_abs: np.ndarray,
              thr_hi: float, thr_abs: float, ref_sd: float, n_boot: int,
              rng: np.random.Generator) -> dict:
    s = group["score"].to_numpy()
    a = np.abs(s)
    ref_sorted, abs_sorted = np.sort(ref_res), np.sort(ref_abs)
    c_hi, c_abs = auc_contrib(s, ref_sorted), auc_contrib(a, abs_sorted)
    fields = group["field_id"].to_numpy()
    uniq = np.unique(fields)
    idx_by_field = [np.where(fields == f)[0] for f in uniq]
    boots = np.empty((n_boot, 2))
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ids = np.concatenate([idx_by_field[p] for p in pick])
        boots[b] = [c_hi[ids].mean(), c_abs[ids].mean()]
    lo_hi, up_hi = np.percentile(boots[:, 0], [2.5, 97.5])
    lo_ab, up_ab = np.percentile(boots[:, 1], [2.5, 97.5])
    return {"n_fields": len(uniq), "n_samples": len(s),
            "mean_score_bits": s.mean(), "mean_score_in_ref_sd": s.mean() / ref_sd,
            "AUC_high": c_hi.mean(), "AUC_high_lo": lo_hi, "AUC_high_hi": up_hi,
            "AUC_abs": c_abs.mean(), "AUC_abs_lo": lo_ab, "AUC_abs_hi": up_ab,
            "TPR_at_5pct_FPR_high": float((s > thr_hi).mean()),
            "TPR_at_5pct_FPR_abs": float((a > thr_abs).mean())}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    if "level" not in df.columns:
        # tolerate other names for the perturbation level column
        for alt in ("value", "param", "parameter", "size", "sigma", "block_size", "factor"):
            if alt in df.columns:
                df["level"] = df[alt]
                break
    need = {"field_id", "axis", "level", "bits_per_interior_dim"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{path}: missing columns {sorted(miss)}; columns found: {list(df.columns)}")
    if "replicate" not in df.columns:
        df["replicate"] = 0
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=Path, default=Path("e1_analysis_table.csv"))
    p.add_argument("--results", type=Path, action="append", required=True,
                   help="a perturbation results jsonl; repeat for several files")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("screening_test_results.csv"))
    args = p.parse_args()

    tab = pd.read_csv(args.table).sort_values("field_id").reset_index(drop=True)
    files = {str(f): load_jsonl(f) for f in args.results}
    selected = sorted(set().union(*[set(d["field_id"].unique()) for d in files.values()]))
    ref = tab[~tab["field_id"].isin(selected)].reset_index(drop=True)
    print(f"Reference real days: {len(ref)} (excluded the {len(selected)} perturbed fields)")

    ref_res = oof_residuals(ref)
    ref_sd = float(ref_res.std(ddof=1))
    ref_abs = np.abs(ref_res)
    thr_hi = float(np.quantile(ref_res, 0.95))
    thr_abs = float(np.quantile(ref_abs, 0.95))
    beta = fit(ref["wet_fraction"].to_numpy(float), ref["season"].to_numpy(),
               ref["bits_per_interior_dim"].to_numpy(float))
    print(f"Reference residual SD (out of fold): {ref_sd:.4f} bits/dim; "
          f"95th percentile {thr_hi:.4f}; 95th percentile of |score| {thr_abs:.4f}\n")

    info = tab.set_index("field_id")
    rng = np.random.default_rng(args.seed)
    out = []
    for name, df in files.items():
        df = df[df["field_id"].isin(info.index)].copy()
        df["wf"] = info.loc[df["field_id"], "wet_fraction"].to_numpy(float)
        df["season"] = info.loc[df["field_id"], "season"].to_numpy()
        df["score"] = df["bits_per_interior_dim"].to_numpy(float) - predict(
            beta, df["wf"].to_numpy(), df["season"].to_numpy())
        # for swap, "level" is the partner day's id: pool all partners into one group
        df["level"] = df["level"].astype(object)
        df.loc[df["axis"] == "swap", "level"] = "all partners"
        df["level"] = df["level"].astype(str)
        for (axis, level), g in df.groupby(["axis", "level"], sort=False):
            r = summarise(g, ref_res, ref_abs, thr_hi, thr_abs, ref_sd, args.n_boot, rng)
            r.update({"file": Path(name).name, "axis": axis, "level": level,
                      "role": "control (unperturbed)" if axis == "baseline" else "perturbed"})
            out.append(r)
    res = pd.DataFrame(out)
    cols = ["file", "axis", "level", "role", "n_fields", "mean_score_bits", "mean_score_in_ref_sd",
            "AUC_high", "AUC_high_lo", "AUC_high_hi", "AUC_abs", "AUC_abs_lo", "AUC_abs_hi",
            "TPR_at_5pct_FPR_high", "TPR_at_5pct_FPR_abs"]
    res = res[cols]
    res.to_csv(args.out, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:0.3f}")
    print("=" * 110)
    print("SCREENING RESULTS (AUC 0.5 = no separation, 1.0 = perfect). CI = field bootstrap.")
    print("=" * 110)
    show = res.copy()
    show["AUC_high [95% CI]"] = show.apply(
        lambda r: f"{r.AUC_high:.2f} [{r.AUC_high_lo:.2f},{r.AUC_high_hi:.2f}]", axis=1)
    show["AUC_abs [95% CI]"] = show.apply(
        lambda r: f"{r.AUC_abs:.2f} [{r.AUC_abs_lo:.2f},{r.AUC_abs_hi:.2f}]", axis=1)
    print(show[["file", "axis", "level", "role", "mean_score_bits", "mean_score_in_ref_sd",
                "AUC_high [95% CI]", "AUC_abs [95% CI]", "TPR_at_5pct_FPR_high"]].to_string(index=False))
    print("\nREADING GUIDE")
    print("1. Control rows (axis 'baseline') should sit near AUC 0.5: real fields are not flagged.")
    print("2. AUC_high asks 'is the perturbed field scored WORSE than real days?'. AUC_abs ignores")
    print("   direction, useful for mild blur where BPD goes DOWN before it goes up.")
    print("3. A large mean score in reference-SD units with AUC near 1 means the screen works for")
    print("   that perturbation; AUC near 0.5 means the response is real but too small to flag")
    print("   against ordinary day-to-day variation.")
    print("4. Only 20 fields: judge by the CI, not the point value. Target blur uses the original")
    print("   field's wet fraction (approximate); shuffle and all conditioning axes are exact.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
