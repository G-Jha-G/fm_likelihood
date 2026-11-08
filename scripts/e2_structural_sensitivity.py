#!/usr/bin/env python3
"""
e2_structural_sensitivity.py
===============================

E2: does the density see spatial structure in the TARGET field at all,
or only bulk statistics? Table 2's "spatial organisation" covariate
group added ~0 incremental R^2, ambiguous on its own: either the
density genuinely doesn't see structure, or those four summary
statistics (RAPS slope, connected components, Moran's I, intensity
concentration) just don't capture whatever structure it does see. Two
perturbation axes on the target field itself (E3 perturbed the
CONDITIONING; this perturbs the TARGET, conditioning stays true
throughout):

  blur          : Gaussian blur, sigma in {0,1,2,4,8,16,32} pixels.
                  Uses the SAME reflect-pad-before-convolve fix
                  verified and applied to E3's blur axis (the original
                  zero-padding version was proven to distort a uniform
                  field's corner by 74% at large sigma; this was never
                  built with the buggy version in the first place).
  block-shuffle : divide the field into non-overlapping BxB blocks,
                  block size in {2,4,8,16,32}, randomly permute pixel
                  VALUES within each block. Exactly conserves each
                  block's local value distribution (so domain-wide
                  statistics like wet fraction are essentially
                  untouched) while destroying spatial arrangement
                  below the block scale, a genuinely different kind of
                  degradation than blur (smooth gradients vs scrambled
                  noise with identical local statistics). Vectorized
                  via the argsort-of-random-values trick (sorting
                  random numbers gives a random permutation; doing
                  this per-block via argsort(dim=1) gives independent
                  permutations for every block in one call, no
                  per-block Python loop, needed since block_size=2 on
                  this domain gives 30,976 blocks).

Reuses E3's exact 20 fields (season x wet-fraction-percentile
stratified), enabling a direct per-field comparison: does structural
sensitivity (E2) correlate with conditioning sensitivity (E3)?

COMMON RANDOM NUMBERS: dequant and Hutchinson generators are shared
across every level of both axes for a given field+replicate (same
principle as E3). The shuffle PERMUTATION itself is necessarily
level-specific (different block sizes need differently-shaped
permutations) and is drawn fresh per (field, replicate, block_size),
so replicates capture genuine shuffle-to-shuffle variation, not just
solver/dequant noise on a single fixed shuffle.

USAGE
-----
    python scripts/e2_structural_sensitivity.py --pilot --device cuda:1
    python scripts/e2_structural_sensitivity.py --device cuda:1
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
BLUR_LEVELS = [1, 2, 4, 8, 16, 32]
SHUFFLE_LEVELS = [2, 4, 8, 16, 32]
N_REPLICATES = 3


def gaussian_kernel_2d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel2d.unsqueeze(0).unsqueeze(0)


def apply_blur(field_mm: torch.Tensor, sigma: float) -> torch.Tensor:
    """Reflect-pads before convolving, matching E3's verified fix and
    this project's domain-boundary convention throughout. This was
    never built with the zero-padding bug, no history to fix here."""
    if sigma == 0:
        return field_mm
    kernel = gaussian_kernel_2d(sigma, field_mm.device, field_mm.dtype)
    pad = kernel.shape[-1] // 2
    padded = F.pad(field_mm, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel, padding=0)


def apply_block_shuffle(field_mm: torch.Tensor, block_size: int,
                         generator: torch.Generator) -> torch.Tensor:
    """Randomly permutes pixel VALUES within each non-overlapping BxB
    block. Exact value conservation within each block, verified: same
    multiset of values before/after, zero cross-block leakage, genuine
    (not identity) reordering, reproducible given the same generator.
    Vectorized: no per-block Python loop.

    generator may live on a different device than field_mm (a CPU
    generator producing randomness for a CUDA tensor, the normal case
    here): generate on the GENERATOR's own device first, matching the
    exact pattern PrecipTransform.dequantize() already uses correctly,
    then move the result to field_mm's device."""
    B = block_size
    _, _, H, W = field_mm.shape
    nH, nW = H // B, W // B
    x = field_mm.reshape(H, W)
    x = x.reshape(nH, B, nW, B).permute(0, 2, 1, 3).reshape(nH * nW, B * B)
    rand_vals = torch.rand(nH * nW, B * B, generator=generator,
                            device=generator.device).to(field_mm.device)
    perm_idx = rand_vals.argsort(dim=1)
    x_shuffled = torch.gather(x, 1, perm_idx)
    x_shuffled = x_shuffled.reshape(nH, nW, B, B).permute(0, 2, 1, 3).reshape(1, 1, H, W)
    return x_shuffled


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
    p.add_argument("--fields-csv", type=Path, default=Path("e3_selected_fields.csv"),
                   help="reuses E3's field selection")
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "best_fm_v1.pt"))
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "e2_outputs")
    p.add_argument("--device", type=str, default=None,
                   help="e.g. cuda:0, cuda:1, cpu. Default: auto. Check nvidia-smi first "
                        "on this shared server.")
    args = p.parse_args()

    if not args.fields_csv.exists():
        print(f"error: {args.fields_csv} not found")
        return 1

    fields_df = pd.read_csv(args.fields_csv)
    if args.pilot:
        fields_df = fields_df.iloc[:1]
        blur_levels = [2, 16]
        shuffle_levels = [2, 32]
        n_reps = 2
        print("PILOT MODE: 1 field, reduced levels, 2 replicates\n")
    else:
        blur_levels, shuffle_levels = BLUR_LEVELS, SHUFFLE_LEVELS
        n_reps = N_REPLICATES

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "e2_results.jsonl"
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
            units.append((fid, rep, "baseline", 0))
            for lvl in blur_levels:
                units.append((fid, rep, "blur", lvl))
            for lvl in shuffle_levels:
                units.append((fid, rep, "shuffle", lvl))

    remaining_units = [u for u in units if (u[0], u[1], u[2], str(u[3])) not in done]
    print(f"Total units: {len(units)}  Already done: {len(units)-len(remaining_units)}  "
          f"Remaining: {len(remaining_units)}\n")

    completed_count, elapsed_total = 0, 0.0
    with open(results_path, "a") as out_fh:
        for field_id, rep, axis, level in remaining_units:
            p_true = p_hr_mm[field_id:field_id + 1].to(device)
            lr_true = lr_mm_all[field_id:field_id + 1].to(device)
            cond_u_true = normalizer.normalize_torch(lr_true)

            if axis == "baseline":
                p_pert = p_true
            elif axis == "blur":
                p_pert = apply_blur(p_true, float(level))
            elif axis == "shuffle":
                shuffle_gen = torch_generator(field_id=field_id, condition="e2_common",
                                               replicate=rep, purpose=f"shuffle_block{level}")
                p_pert = apply_block_shuffle(p_true, int(level), shuffle_gen)
            else:
                raise ValueError(axis)

            dq_gen = torch_generator(field_id=field_id, condition="e2_common",
                                      replicate=rep, purpose="dequant")
            hu_gen = torch_generator(field_id=field_id, condition="e2_common",
                                      replicate=rep, purpose="hutchinson")

            t0 = time.time()
            out = bits_per_dim(engine, transform, p_pert, cond_u_true, dq_gen, hu_gen,
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
