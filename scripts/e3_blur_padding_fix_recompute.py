#!/usr/bin/env python3
"""
e3_blur_padding_fix_recompute.py
===================================

Fixes a confirmed bug: the original apply_blur used F.conv2d's default
zero-padding for the extra border a large Gaussian kernel needs beyond
the field's own 1px reflect-padded edge. Proven directly on a uniform
test field (any correct blur must leave it completely unchanged): the
old version pulled the corner value from 10.0 down to 2.56, a 74%
distortion from phantom zeros, not signal. At E3's tested sigma=8
(kernel radius 24px), roughly 25% of the 352x352 domain had some degree
of this contamination.

This recomputes ONLY the blur axis (levels 1, 2, 4, 8), reusing the
existing baseline values unchanged: the baseline never calls apply_blur
at all (it's the unperturbed field), so it was never affected by the
bug and recomputing it would be wasted GPU time. Everything else
(field selection, common-random-number generator keys, replicate
structure) is identical to the original E3 run, so the comparison
between old (buggy) and new (fixed) blur results isolates exactly the
effect of the padding fix, nothing else changed.

Output is a self-contained file (baseline rows copied in alongside the
newly computed blur rows) so analyze_e3_blur_corrected.py works on it
directly with no modification, just point --results at the new file.

USAGE
-----
    python scripts/e3_blur_padding_fix_recompute.py --pilot --device cuda:1
    python scripts/e3_blur_padding_fix_recompute.py --device cuda:1
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
BLUR_LEVELS = [1, 2, 4, 8]
N_REPLICATES = 3


def gaussian_kernel_2d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel2d.unsqueeze(0).unsqueeze(0)


def apply_blur_fixed(lr_mm: torch.Tensor, sigma: float) -> torch.Tensor:
    """FIXED: reflect-pads before convolving, matching this project's
    established domain-boundary convention, instead of relying on
    conv2d's zero-padding default."""
    if sigma == 0:
        return lr_mm
    kernel = gaussian_kernel_2d(sigma, lr_mm.device, lr_mm.dtype)
    pad = kernel.shape[-1] // 2
    padded = F.pad(lr_mm, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel, padding=0)


def load_all_fields(device: str, checkpoint_path: str):
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
        hr = f["precipitation"][:].astype(np.float32)
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
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--fields-csv", type=Path, default=Path("e3_selected_fields.csv"))
    p.add_argument("--old-results", type=Path, default=Path("e3_outputs/e3_results.jsonl"),
                   help="original E3 output, used ONLY to copy over baseline rows "
                        "(unaffected by the bug, not recomputed)")
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "best_fm_v1.pt"))
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "e3_blur_fixed_outputs")
    p.add_argument("--device", type=str, default=None,
                   help="e.g. cuda:0, cuda:1, cpu. Default: auto (cuda:0 if available, else cpu). "
                        "Use this to pick a GPU with free memory on a shared server.")
    args = p.parse_args()

    if not args.fields_csv.exists() or not args.old_results.exists():
        print(f"error: need both {args.fields_csv} and {args.old_results}")
        return 1

    fields_df = pd.read_csv(args.fields_csv)
    if args.pilot:
        fields_df = fields_df.iloc[:1]
        blur_levels = [2, 8]
        n_reps = 2
        print("PILOT MODE: 1 field, reduced levels, 2 replicates\n")
    else:
        blur_levels = BLUR_LEVELS
        n_reps = N_REPLICATES

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "e3_blur_fixed_results.jsonl"

    existing_lines = set()
    if results_path.exists():
        with open(results_path) as fh:
            for line in fh:
                r = json.loads(line)
                existing_lines.add((r["field_id"], r["replicate"], r["axis"]))

    field_ids = set(int(f) for f in fields_df["field_id"])
    baseline_copied = 0
    with open(results_path, "a") as out_fh:
        with open(args.old_results) as old_fh:
            for line in old_fh:
                r = json.loads(line)
                if (r["axis"] == "baseline" and r["field_id"] in field_ids
                        and (r["field_id"], r["replicate"], "baseline") not in existing_lines):
                    if args.pilot and r["field_id"] != int(fields_df.iloc[0]["field_id"]):
                        continue
                    out_fh.write(json.dumps(r) + "\n")
                    baseline_copied += 1
    print(f"Copied {baseline_copied} baseline rows from {args.old_results} (unaffected, reused)")

    done = load_completed(results_path)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
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
            for lvl in blur_levels:
                units.append((fid, rep, "blur", lvl))

    remaining_units = [u for u in units if (u[0], u[1], u[2], str(u[3])) not in done]
    print(f"Total blur units: {len(units)}  Already done: {len(units)-len(remaining_units)}  "
          f"Remaining: {len(remaining_units)}\n")

    completed_count, elapsed_total = 0, 0.0
    with open(results_path, "a") as out_fh:
        for field_id, rep, axis, level in remaining_units:
            p_field = p_hr_mm[field_id:field_id + 1].to(device)
            lr_true = lr_mm_all[field_id:field_id + 1].to(device)

            lr_pert = apply_blur_fixed(lr_true, float(level))
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
                  f"blur={level}  bpd={record['bits_per_interior_dim']:.4f}  {dt:.1f}s  "
                  f"ETA {eta/60:.1f} min", flush=True)

    print(f"\nDone. New units this run: {completed_count}, "
          f"elapsed: {elapsed_total/60:.1f} min")
    print(f"\nNext: python scripts/analyze_e3_blur_corrected.py "
          f"--results {results_path}")


if __name__ == "__main__":
    main()
