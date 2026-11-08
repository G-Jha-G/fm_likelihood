#!/usr/bin/env python3
"""
e3_wetfraction_confound_check.py
====================================
Review concern #2 (major): Sect 4.2 (E1) explicitly controls for the
wet-fraction confound before reporting BPD-vs-extremity correlations.
Sects 4.3-4.4 (E3)'s magnitude and blur perturbation axes do NOT get the
same treatment, even though both perturbations plausibly move the
conditioning field's IMPLIED wetness away from the fixed target's actual
wetness:

  magnitude: scaling the coarse field by `level` directly rescales its
             wet-area intensity and, for a zero-inflated field, its
             above-threshold wet fraction too (scaling never changes
             which cells are > 0, but it does change how many cells
             clear the 1 mm wet-definition threshold).
  blur     : Gaussian-blurring a zero-inflated field spreads mass from
             wet cells into dry neighbors, which can only ever INCREASE
             the blurred field's wet fraction at a fixed threshold
             (never decrease it) -- exactly the mechanism review concern
             #2 names for "target-blur . . . converts some exact zeros
             to small positive values".

If most of magnitude/blur's dose-response correlation with BPD is
actually explained by this wet-fraction mismatch (the same confound
Sect 4.2 worried about) rather than by "the model noticing the
conditioning is wrong" in some richer sense, that is worth knowing
before reporting rho=0.86-0.99 as evidence of fine-grained conditioning
sensitivity.

WHAT "MISMATCH" MEANS HERE
-----------------------------
mismatch(level) = wet_fraction(perturbed coarse field, threshold=1mm)
                   - wet_fraction(TARGET field, from e3_selected_fields.csv)

computed on the perturbed LR conditioning field used by
scripts/e3_conditioning_sensitivity.py itself (same avg_pool2d+bicubic
compute_lr_fullfield, same apply_blur, same MAGNITUDE_LEVELS/BLUR_LEVELS),
so this is the exact quantity the trained model was actually shown, not
an approximation of it.

STATISTICS: per-field partial Spearman, pooling replicates (same
convention as analyze_e3_conditioning_sensitivity.py), of BPD vs
distance CONTROLLING for mismatch, and BPD vs mismatch CONTROLLING for
distance, then an iid field-bootstrap 95% CI across the 20 fields on
each partial correlation (same bootstrap analyze_e3_conditioning_
sensitivity.py already uses). If the raw dose-response correlation
survives controlling for mismatch nearly unchanged, the confound is not
doing the work; if it collapses toward zero while the mismatch-given-
distance correlation stays strong, a meaningful share of the reported
sensitivity is the wet-fraction confound, not novel conditioning
sensitivity.

REQUIRES GPU? No. This script only needs the raw MSWX h5 (to recompute
the coarse conditioning field exactly as the production pipeline does)
and train_precip_fm_v1.py's compute_lr_fullfield; it never loads the
trained checkpoint or runs the ODE. BPD values are read from an existing
E3 results file (e3_results_FINAL_corrected.jsonl or a fresh
e3_outputs/e3_results.jsonl from scripts/e3_conditioning_sensitivity.py)
rather than recomputed.

USAGE
-----
    python e3_wetfraction_confound_check.py \\
        --results e3_results_FINAL_corrected.jsonl \\
        --fields-csv e3_selected_fields.csv
"""
from __future__ import annotations

import os
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_common import field_bootstrap_dist, ols_residuals, percentile_ci, rankdata  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MSWX_TEST_H5 = os.environ.get("MSWX_TEST_H5", "data/mswx_test_2011-2014.h5")
WET_THRESHOLD_MM = 1.0

MAGNITUDE_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
BLUR_LEVELS = [0, 1, 2, 4, 8]

DISTANCE_FN = {
    "magnitude": lambda level: abs(np.log2(level)),
    "blur": lambda level: float(level),
}


def gaussian_kernel_2d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel2d.unsqueeze(0).unsqueeze(0)


def apply_blur(lr_mm: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma == 0:
        return lr_mm
    kernel = gaussian_kernel_2d(sigma, lr_mm.device, lr_mm.dtype)
    pad = kernel.shape[-1] // 2
    # Reflect-pad BEFORE convolving, then convolve with no implicit padding.
    # The production scripts/e3_conditioning_sensitivity.py originally used
    # F.conv2d(..., padding=pad), i.e. ZERO padding, which pulls border values
    # toward zero (by up to 74 percent at sigma=8 on a uniform test field).
    # The corrected pipeline (manuscript Sect. 4.3) uses reflect padding, so the
    # mismatch table must be built the same way to line up with the corrected BPD.
    padded = F.pad(lr_mm, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel, padding=0)


def wet_fraction(field_mm: torch.Tensor, threshold: float = WET_THRESHOLD_MM) -> float:
    return float((field_mm > threshold).float().mean().item())


def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_mismatch_table(fields_df: pd.DataFrame, h5_path: str) -> pd.DataFrame:
    """For every selected field and every magnitude/blur level, the
    coarse-conditioning wet fraction under that perturbation, and its
    mismatch against the target's own (interior, from the CSV) wet
    fraction. CPU-only; no model, no GPU."""
    import importlib.util
    train_script = REPO_ROOT / "train_precip_fm_v1.py"
    if not train_script.exists():
        # fall back to wherever it was uploaded alongside this project
        alt = Path("/mnt/user-data/uploads/train_precip_fm_v1.py")
        train_script = alt if alt.exists() else train_script
    spec = importlib.util.spec_from_file_location("train_precip_fm_v1", train_script)
    tpf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpf)

    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        hr_all = f["precipitation"][:].astype(np.float32)
    H, W = hr_all.shape[1], hr_all.shape[2]

    rows = []
    for _, frow in fields_df.iterrows():
        fid = int(frow["field_id"])
        target_wf = float(frow["wet_fraction"])
        hr = hr_all[fid:fid + 1]
        lr_true_np = tpf.compute_lr_fullfield(hr, tpf.Config().scale_factor, H, W)
        lr_true = torch.from_numpy(lr_true_np).unsqueeze(1)

        # baseline point (level=1.0 magnitude / level=0 blur are the SAME
        # underlying unperturbed coarse field; include once under each axis
        # so every axis has its own distance=0 anchor, matching
        # analyze_e3_conditioning_sensitivity.py's per_field_dose_response)
        base_wf = wet_fraction(lr_true)
        rows.append({"field_id": fid, "axis": "magnitude", "level": 1.0,
                      "coarse_wet_fraction": base_wf, "mismatch": base_wf - target_wf})
        rows.append({"field_id": fid, "axis": "blur", "level": 0,
                      "coarse_wet_fraction": base_wf, "mismatch": base_wf - target_wf})

        for lvl in MAGNITUDE_LEVELS:
            if lvl == 1.0:
                continue
            perturbed = lr_true * lvl
            wf = wet_fraction(perturbed)
            rows.append({"field_id": fid, "axis": "magnitude", "level": lvl,
                          "coarse_wet_fraction": wf, "mismatch": wf - target_wf})

        for lvl in BLUR_LEVELS:
            if lvl == 0:
                continue
            perturbed = apply_blur(lr_true, float(lvl))
            wf = wet_fraction(perturbed)
            rows.append({"field_id": fid, "axis": "blur", "level": lvl,
                          "coarse_wet_fraction": wf, "mismatch": wf - target_wf})

    out = pd.DataFrame(rows)
    out["blur_padding"] = "reflect"  # marks tables built with the corrected blur
    return out


def per_field_partial_correlations(bpd_rows: list[dict], mismatch_df: pd.DataFrame,
                                     axis: str) -> dict:
    """field_id -> dict(raw_rho_distance, partial_rho_distance_given_mismatch,
    partial_rho_mismatch_given_distance), pooling replicates, mirroring
    analyze_e3_conditioning_sensitivity.py's per_field_dose_response but
    adding mismatch as a second regressor."""
    mismatch_lookup = {(r["field_id"], r["level"]): r["mismatch"]
                        for _, r in mismatch_df[mismatch_df["axis"] == axis].iterrows()}

    by_field = defaultdict(list)  # field_id -> list of (distance, mismatch, bpd)
    for r in bpd_rows:
        if r["axis"] == axis:
            level = r["level"]
        elif r["axis"] == "baseline" and axis == "magnitude":
            level = 1.0
        elif r["axis"] == "baseline" and axis == "blur":
            level = 0
        else:
            continue
        key = (r["field_id"], level)
        if key not in mismatch_lookup:
            continue
        d = DISTANCE_FN[axis](level)
        m = mismatch_lookup[key]
        by_field[r["field_id"]].append((d, m, r["bits_per_interior_dim"]))

    out = {}
    for field_id, triples in by_field.items():
        if len(triples) < 4:
            continue
        d = np.array([t[0] for t in triples], dtype=np.float64)
        m = np.array([abs(t[1]) for t in triples], dtype=np.float64)  # fold: magnitude of mismatch
        y = np.array([t[2] for t in triples], dtype=np.float64)

        if len(np.unique(d)) < 2 or len(np.unique(y)) < 2:
            continue
        rd, rm, ry = rankdata(d), rankdata(m), rankdata(y)

        # raw
        rd_c, ry_c = rd - rd.mean(), ry - ry.mean()
        denom = float(np.sqrt((rd_c ** 2).sum() * (ry_c ** 2).sum()))
        raw_rho = float((rd_c * ry_c).sum() / denom) if denom > 0 else float("nan")

        # partial: y vs d | m
        if len(np.unique(m)) >= 2:
            Xm = np.column_stack([np.ones(len(d)), rm])
            y_resid = ols_residuals(ry, Xm)
            d_resid = ols_residuals(rd, Xm)
            denom2 = float(np.sqrt((y_resid ** 2).sum() * (d_resid ** 2).sum()))
            partial_d_given_m = float((y_resid * d_resid).sum() / denom2) if denom2 > 0 else float("nan")

            Xd = np.column_stack([np.ones(len(d)), rd])
            y_resid2 = ols_residuals(ry, Xd)
            m_resid = ols_residuals(rm, Xd)
            denom3 = float(np.sqrt((y_resid2 ** 2).sum() * (m_resid ** 2).sum()))
            partial_m_given_d = float((y_resid2 * m_resid).sum() / denom3) if denom3 > 0 else float("nan")
        else:
            partial_d_given_m = raw_rho
            partial_m_given_d = float("nan")

        out[field_id] = {"raw_rho_distance": raw_rho,
                          "partial_rho_distance_given_mismatch": partial_d_given_m,
                          "partial_rho_mismatch_given_distance": partial_m_given_d,
                          "n_points": len(triples)}
    return out


def summarize(name: str, values: dict, n_boot: int, rng: np.random.Generator):
    vals = np.array([v for v in values.values() if not np.isnan(v)])
    if len(vals) == 0:
        print(f"  {name}: no valid field-level estimates")
        return
    point = float(vals.mean())
    boot = field_bootstrap_dist(vals, n_boot, rng)
    lo, hi = percentile_ci(boot)
    excl = "EXCLUDES zero" if (lo > 0 or hi < 0) else "includes zero"
    print(f"  {name:>42}: mean={point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"(n_fields={len(vals)})  {excl}")


def blur_paired_mismatch_control(bpd_rows: list[dict], mismatch_df: pd.DataFrame,
                                  n_boot: int, rng: np.random.Generator,
                                  levels=(1, 2, 4, 8)) -> None:
    """Mismatch-controlled version of the mild-vs-aggressive blur contrast.

    Blur is non-monotonic (mild blur lowers BPD, sigma=8 raises it), so a
    single rank correlation with sigma is the wrong summary. For each blur
    level L and each field:
        effect_f(L)  = mean BPD(blur=L) - mean BPD(baseline)    (over replicates)
        dmis_f(L)    = |mismatch(L)| - |mismatch(0)|            (change in the
                       absolute gap between the conditioning's wet fraction
                       and the target's wet fraction caused by the blur)
    then, ACROSS fields, regress effect on dmis. The slope says how much of
    the effect travels with wet-fraction mismatch; the INTERCEPT is the
    estimated effect of the blur for a field whose wet-fraction mismatch did
    not change at all, i.e. the part that is not the Sect 4.2 wet-fraction
    mechanism. Field-level iid bootstrap (resample fields, refit).
    """
    base = defaultdict(list)
    byl = defaultdict(list)
    for r in bpd_rows:
        if r["axis"] == "baseline":
            base[r["field_id"]].append(r["bits_per_interior_dim"])
        elif r["axis"] == "blur":
            byl[(r["field_id"], float(r["level"]))].append(r["bits_per_interior_dim"])
    mis = {(int(r["field_id"]), float(r["level"])): abs(float(r["mismatch"]))
           for _, r in mismatch_df[mismatch_df["axis"] == "blur"].iterrows()}

    print(f"{'sigma':>6} {'n':>3} {'mean effect':>12} {'mean d|mis|':>12} "
          f"{'sd d|mis|':>10} {'slope':>9} {'intercept (95% CI)':>34}")
    for L in levels:
        eff, dm = [], []
        for fid, b in base.items():
            k = (fid, float(L))
            if k in byl and k in mis and (fid, 0.0) in mis:
                eff.append(np.mean(byl[k]) - np.mean(b))
                dm.append(mis[k] - mis[(fid, 0.0)])
        eff, dm = np.array(eff), np.array(dm)
        if len(eff) < 5:
            print(f"{L:>6} {len(eff):>3}  too few fields")
            continue

        def fit(e, d):
            if np.std(d) < 1e-9:
                return float("nan"), float(np.mean(e))
            slope, icpt = np.polyfit(d, e, 1)
            return float(slope), float(icpt)

        slope, icpt = fit(eff, dm)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, len(eff), size=len(eff))
            boots[b] = fit(eff[idx], dm[idx])[1]
        lo, hi = percentile_ci(boots)
        print(f"{L:>6} {len(eff):>3} {eff.mean():>+12.4f} {dm.mean():>+12.4f} "
              f"{dm.std():>10.4f} {slope:>+9.3f} {icpt:>+12.4f} [{lo:+.4f}, {hi:+.4f}]")
    print()
    print("Reading: if the intercept at sigma=4 stays negative and at sigma=8 stays positive")
    print("with CIs excluding zero, the mild-blur dip and the aggressive-blur rise are not")
    print("just the wet-fraction mismatch mechanism. A small sd of d|mis| means mismatch barely")
    print("varies across fields at that sigma, so the slope is weakly identified and the")
    print("intercept is then close to the plain mean effect.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("e3_results_FINAL_corrected.jsonl"))
    p.add_argument("--fields-csv", type=Path, default=Path("e3_selected_fields.csv"))
    p.add_argument("--h5", type=str, default=MSWX_TEST_H5)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mismatch-cache", type=Path, default=Path("e3_wetfraction_mismatch.csv"),
                   help="cache the (field, axis, level) -> mismatch table here; "
                        "reused on a re-run instead of recomputing from the h5")
    p.add_argument("--force-recompute", action="store_true",
                   help="ignore any cached mismatch table and rebuild it from the h5")
    args = p.parse_args()

    if not args.results.exists():
        print(f"error: {args.results} not found")
        return 1
    if not args.fields_csv.exists():
        print(f"error: {args.fields_csv} not found")
        return 1

    fields_df = pd.read_csv(args.fields_csv)
    bpd_rows = load_results(args.results)
    print(f"Loaded {len(bpd_rows)} BPD rows, {len(fields_df)} selected fields\n")

    mismatch_df = None
    if args.mismatch_cache.exists() and not args.force_recompute:
        cached = pd.read_csv(args.mismatch_cache)
        if "blur_padding" in cached.columns:
            print(f"Reusing cached mismatch table {args.mismatch_cache}")
            mismatch_df = cached
        else:
            print(f"Cached {args.mismatch_cache} predates the reflect-padding blur fix "
                  f"(no 'blur_padding' column); rebuilding it.")
    if mismatch_df is None:
        mismatch_df = compute_mismatch_table(fields_df, args.h5)
        mismatch_df.to_csv(args.mismatch_cache, index=False)
        print(f"Wrote {args.mismatch_cache}")

    rng = np.random.default_rng(args.seed)

    for axis in ("magnitude", "blur"):
        print()
        print("=" * 96)
        print(f"AXIS: {axis}")
        print("=" * 96)
        per_field = per_field_partial_correlations(bpd_rows, mismatch_df, axis)
        print(f"({len(per_field)} fields with usable data)\n")
        summarize("raw Spearman(BPD, distance)",
                   {k: v["raw_rho_distance"] for k, v in per_field.items()}, args.n_boot, rng)
        summarize("partial Spearman(BPD, distance | |mismatch|)",
                   {k: v["partial_rho_distance_given_mismatch"] for k, v in per_field.items()},
                   args.n_boot, rng)
        summarize("partial Spearman(BPD, |mismatch| | distance)",
                   {k: v["partial_rho_mismatch_given_distance"] for k, v in per_field.items()},
                   args.n_boot, rng)

    print()
    print("=" * 96)
    print("BLUR, PAIRED CONTRAST WITH MISMATCH CONTROL (per blur level, across fields)")
    print("=" * 96)
    blur_paired_mismatch_control(bpd_rows, mismatch_df, args.n_boot, rng)

    print()
    print("=" * 96)
    print("READING GUIDE")
    print("=" * 96)
    print("If 'partial Spearman(BPD, distance | mismatch)' stays close to the RAW correlation")
    print("reported in the manuscript, the dose-response finding is not primarily the")
    print("wet-fraction confound. If it collapses toward zero while 'partial Spearman(BPD,")
    print("mismatch | distance)' stays clearly nonzero, report the magnitude/blur sensitivity")
    print("as substantially explained by the same wet-fraction mechanism already controlled")
    print("for in Sect 4.2, not as independent evidence of fine-grained conditioning sensitivity.")


if __name__ == "__main__":
    raise SystemExit(main())
