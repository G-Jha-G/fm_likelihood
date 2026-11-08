#!/usr/bin/env python3
"""
e3_conditioning_sensitivity.py
=================================

E3: does the density actually respond to its conditioning, or is
"conditional likelihood" doing less work than the term implies? Four
perturbation axes, each testing a different way the conditioning could
matter:

  magnitude : scale the raw (mm) coarse field by a factor, {0.25,0.5,
              0.75,1,1.5,2,4}x. Tests sensitivity to HOW MUCH rain the
              model was told to expect.
  shift     : circularly shift the raw coarse field by {0,1,2,4,8,16,32}
              pixels (both spatial axes). Tests sensitivity to WHERE.
  blur      : Gaussian-blur the raw coarse field, sigma in
              {0,1,2,4,8} pixels. Tests sensitivity to the coarse
              field's spatial PATTERN, distinct from its magnitude.
  swap      : replace the conditioning entirely with a real different
              day's, wet-fraction-matched (see e3_swap_partners.csv,
              caliper 0.03, minimum 30-day gap, maximizing spectral
              difference among candidates). The realistic version of
              "wrong conditioning": an actual synoptic pattern, not a
              synthetic distortion.

CRITICAL: magnitude and blur are applied to the RAW mm conditioning
field, BEFORE normalisation, not to the already-normalised network
input. Normalisation is a nonlinear (log1p-based) pointwise transform;
scaling the normalised value by 2x does NOT correspond to "twice as
much rain". Shift is applied on the raw field too, for consistency,
though it would give an identical result applied after normalisation
(shifting commutes with a pointwise transform, scaling does not).

COMMON RANDOM NUMBERS: for a given field and replicate, the SAME
dequant and Hutchinson generator keys are used across every axis and
level (baseline shared across magnitude=1x, shift=0, blur=0), so any
BPD difference reflects the perturbation, not resampling noise. The
target (high-resolution) field is NEVER perturbed, only its
conditioning; log_prob_model and log_jac's dequantisation both depend
only on the target, so only log_prob_model can change here, isolating
exactly the effect this experiment is designed to measure.

USAGE
-----
    python scripts/e3_conditioning_sensitivity.py --pilot   # 1 field, 2 levels/axis
    python scripts/e3_conditioning_sensitivity.py            # full: 20 fields, all levels
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train_precip_fm_v1 import Config, PrecipNormalizer, PrecipUNet, compute_lr_fullfield

from fm_likelihood.bpd import bits_per_dim
from fm_likelihood.cnf import ConditionalCNFLikelihood, LikelihoodConfig
from fm_likelihood.mask import compute_padding
from fm_likelihood.seeds import torch_generator
from fm_likelihood.transform import PrecipConfig, PrecipTransform
from fm_likelihood.velocity import FMVelocity

MSWX_TEST_H5 = os.environ.get("MSWX_TEST_H5", "data/mswx_test_2011-2014.h5")
MAGNITUDE_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
SHIFT_LEVELS = [0, 1, 2, 4, 8, 16, 32]
BLUR_LEVELS = [0, 1, 2, 4, 8]
N_REPLICATES = 3


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
    # reflect-pad before convolving; conv2d padding=pad would zero-pad the border
    return F.conv2d(F.pad(lr_mm, (pad, pad, pad, pad), mode="reflect"), kernel, padding=0)


def apply_shift(lr_mm: torch.Tensor, n_pixels: int) -> torch.Tensor:
    if n_pixels == 0:
        return lr_mm
    return torch.roll(lr_mm, shifts=(n_pixels, n_pixels), dims=(-2, -1))


def load_all_fields(device: str, checkpoint_path: str):
    """Loads everything needed, including the RAW (mm) LR conditioning
    field, not just the normalised version, since magnitude/blur
    perturbations must happen before normalisation."""
    cfg_train = Config()
    model = PrecipUNet(in_channels=2, out_channels=1, base_dim=cfg_train.base_dim,
                        ch_mult_cap=cfg_train.ch_mult_cap, time_dim=cfg_train.time_dim
                        ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()
    velocity = FMVelocity(model, time_scale=cfg_train.t_scale)
    normalizer = PrecipNormalizer(cfg_train.log1p_max)

    with h5py.File(MSWX_TEST_H5, "r") as f:
        hr = f["precipitation"][:].astype(np.float32)  # ALL 1461 days, matches e1's indexing
    T, H, W = hr.shape
    info = compute_padding(H, W, div=16)
    lr = compute_lr_fullfield(hr, cfg_train.scale_factor, H, W)

    p_hr_mm = torch.from_numpy(hr).unsqueeze(1)
    lr_mm = torch.from_numpy(lr).unsqueeze(1)
    p_hr_mm = F.pad(p_hr_mm, (info.pad_left, info.pad_right, info.pad_top, info.pad_bottom),
                     mode="reflect")
    lr_mm = F.pad(lr_mm, (info.pad_left, info.pad_right, info.pad_top, info.pad_bottom),
                  mode="reflect")

    return velocity, p_hr_mm, lr_mm, info, normalizer


def load_completed(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["field_id"], r["replicate"], r["axis"], str(r["level"])))
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", action="store_true",
                   help="1 field, 2 levels/axis, no swap partner needed beyond the pair "
                        "already selected. Run this first.")
    p.add_argument("--fields-csv", type=Path, default=Path("e3_selected_fields.csv"))
    p.add_argument("--partners-csv", type=Path, default=Path("e3_swap_partners.csv"))
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "best_fm_v1.pt"))
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "e3_outputs")
    args = p.parse_args()

    if not args.fields_csv.exists() or not args.partners_csv.exists():
        print(f"error: need both {args.fields_csv} and {args.partners_csv}")
        return 1

    fields_df = pd.read_csv(args.fields_csv)
    partners_df = pd.read_csv(args.partners_csv).set_index("field_id")

    if args.pilot:
        fields_df = fields_df.iloc[:1]
        mag_levels = [0.5, 1.0, 2.0]
        shift_levels = [0, 8]
        blur_levels = [0, 4]
        n_reps = 2
        print("PILOT MODE: 1 field, reduced levels, 2 replicates\n")
    else:
        mag_levels, shift_levels, blur_levels = MAGNITUDE_LEVELS, SHIFT_LEVELS, BLUR_LEVELS
        n_reps = N_REPLICATES

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "e3_results.jsonl"
    done = load_completed(results_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model and data on {device}...")
    velocity, p_hr_mm, lr_mm_all, info, normalizer = load_all_fields(device, args.checkpoint)

    transform = PrecipTransform(PrecipConfig())
    engine = ConditionalCNFLikelihood(
        velocity, LikelihoodConfig(method="dopri5", rtol=1e-4, atol=1e-4,
                                    exact_trace=False, hutchinson_probes=1, use_float64=True))

    units = []
    for _, row in fields_df.iterrows():
        fid = int(row["field_id"])
        for rep in range(n_reps):
            units.append((fid, rep, "baseline", 1.0))
            for lvl in mag_levels:
                if lvl == 1.0:
                    continue
                units.append((fid, rep, "magnitude", lvl))
            for lvl in shift_levels:
                if lvl == 0:
                    continue
                units.append((fid, rep, "shift", lvl))
            for lvl in blur_levels:
                if lvl == 0:
                    continue
                units.append((fid, rep, "blur", lvl))
            units.append((fid, rep, "swap", int(partners_df.loc[fid, "partner_id"])))

    remaining_units = [u for u in units if (u[0], u[1], u[2], str(u[3])) not in done]
    print(f"Total units: {len(units)}  Already done: {len(units)-len(remaining_units)}  "
          f"Remaining: {len(remaining_units)}\n")

    completed_count, elapsed_total = 0, 0.0
    with open(results_path, "a") as out_fh:
        for field_id, rep, axis, level in remaining_units:
            p_field = p_hr_mm[field_id:field_id + 1].to(device)
            lr_true = lr_mm_all[field_id:field_id + 1].to(device)

            if axis in ("baseline", "magnitude"):
                lr_pert = lr_true * (level if axis == "magnitude" else 1.0)
            elif axis == "shift":
                lr_pert = apply_shift(lr_true, int(level))
            elif axis == "blur":
                lr_pert = apply_blur(lr_true, float(level))
            elif axis == "swap":
                partner_id = int(level)
                lr_pert = lr_mm_all[partner_id:partner_id + 1].to(device)
            else:
                raise ValueError(axis)

            cond_u = normalizer.normalize_torch(lr_pert)

            dq_gen = torch_generator(field_id=field_id, condition="e3_common",
                                      replicate=rep, purpose="dequant")
            hu_gen = torch_generator(field_id=field_id, condition="e3_common",
                                      replicate=rep, purpose="hutchinson")

            t0 = time.time()
            out = bits_per_dim(engine, transform, p_field, cond_u, dq_gen, hu_gen,
                                d_interior=info.d_interior)
            dt = time.time() - t0
            completed_count += 1
            elapsed_total += dt

            record = {"field_id": field_id, "replicate": rep, "axis": axis, "level": level,
                      "bits_per_interior_dim": out["bits_per_interior_dim"].item(),
                      "n_function_evals": out["n_function_evals"], "elapsed_sec": dt}
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()

            eta = (elapsed_total / completed_count) * (len(remaining_units) - completed_count)
            print(f"[{completed_count}/{len(remaining_units)}] field={field_id} rep={rep} "
                  f"{axis}={level}  bpd={record['bits_per_interior_dim']:.4f}  {dt:.1f}s  "
                  f"ETA {eta/60:.1f} min", flush=True)

    print(f"\nDone. New units this run: {completed_count}, "
          f"elapsed: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
