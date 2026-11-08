#!/usr/bin/env python3
"""
sanity_check_vs_pilot.py
=========================

Confirms the refactored fm_likelihood package reproduces the pilot's own
saved results on real fields, using the real trained network. This is the
first integration test that touches your actual checkpoint and data; it
cannot be verified anywhere except your server, so it exists specifically
to catch anything in the port that the closed-form toy tests couldn't see:
arithmetic that's correct in isolation but wired up wrong here, or drift
in the exact reload path.

WHAT THIS CHECKS
----------------
For the first N test fields (same stride-10 order load_test_fields uses,
so indices line up with the saved npz), compute bits_per_dim through the
NEW package (transform, cnf, velocity, bpd, mask, seeds) and compare
against the OLD pilot's already-saved lve_bpd values in
likelihood_extremes.npz for the SAME field indices.

Exact agreement is NOT expected: the old run used a shared, stateful
generator whose exact draws cannot be reproduced (that statefulness is
the whole reason the refactor exists). This script uses rk4 with the SAME
n_steps and probe count as the pilot's config, for a fair comparison, but
with fresh, independent Hutchinson draws from fm_likelihood.seeds.

What IS expected: agreement within roughly 3 to 5x E0a's own numerical
noise floor (0.0030 bits/dim per field at this exact config), i.e. within
about 0.01 to 0.02 bits/dim. Anything larger points to a real bug in the
port, not sampling noise, since E0a already measured the noise floor at
this probe count and step count directly.

USAGE
-----
    python scripts/sanity_check_vs_pilot.py --n-fields 5

Run from the Likelihood_paper/ root, same place everything else runs
from. Needs the real checkpoint, the real MSWX h5 file, and
train_precip_fm_v1.py all in their existing locations; nothing here
changes any of those paths.
"""
from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
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
def load_model_and_data(device: str, checkpoint_path: str):
    """Mirrors fm_likelihood_extremes_v2.py's run_full() and
    load_test_fields() exactly (re-verified against the source directly
    before writing this, not from memory), so the comparison below is
    apples to apples."""
    cfg_train = Config()

    model = PrecipUNet(
        in_channels=2,
        out_channels=1,
        base_dim=cfg_train.base_dim,
        ch_mult_cap=cfg_train.ch_mult_cap,
        time_dim=cfg_train.time_dim,
    ).to(device)

    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()

    velocity = FMVelocity(model, time_scale=cfg_train.t_scale)

    normalizer = PrecipNormalizer(cfg_train.log1p_max)
    print(f"Loading test fields from: {MSWX_TEST_H5}")
    with h5py.File(MSWX_TEST_H5, "r") as f:
        hr = f["precipitation"][::10].astype(np.float32)

    T, H, W = hr.shape
    info = compute_padding(H, W, div=16)

    lr = compute_lr_fullfield(hr, cfg_train.scale_factor, H, W)
    p_hr_mm = torch.from_numpy(hr).unsqueeze(1)
    lr_t = torch.from_numpy(lr).unsqueeze(1)

    p_hr_mm = F.pad(p_hr_mm, (info.pad_left, info.pad_right, info.pad_top, info.pad_bottom),
                     mode="reflect")
    lr_t = F.pad(lr_t, (info.pad_left, info.pad_right, info.pad_top, info.pad_bottom),
                 mode="reflect")
    cond_u = normalizer.normalize_torch(lr_t)

    return velocity, p_hr_mm, cond_u, info


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-fields", type=int, default=5,
                   help="how many of the first test fields to check")
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "best_fm_v1.pt"))
    p.add_argument("--npz", type=Path, default=REPO_ROOT / "likelihood_extremes.npz",
                   help="the pilot's saved output, for comparison")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-steps", type=int, default=100,
                   help="RK4 steps, matched to the pilot's config (100) for a fair comparison")
    p.add_argument("--probes", type=int, default=8,
                   help="Hutchinson probes, matched to the pilot's config (8)")
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"error: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not args.npz.exists():
        print(f"error: pilot npz not found at {args.npz}", file=sys.stderr)
        sys.exit(1)

    old = np.load(args.npz)
    old_bpd = np.asarray(old["lve_bpd"], dtype=np.float64)

    print(f"Loading model and data on {args.device}...")
    velocity, p_hr_mm, cond_u, info = load_model_and_data(args.device, args.checkpoint)
    print(f"Loaded {p_hr_mm.shape[0]} test fields, padded shape {tuple(p_hr_mm.shape[-2:])}")

    if p_hr_mm.shape[0] != len(old_bpd):
        print(f"[warn] field count mismatch: loaded {p_hr_mm.shape[0]} fields, npz has "
              f"{len(old_bpd)}. Comparing only the first "
              f"{min(p_hr_mm.shape[0], len(old_bpd))} and assuming the ordering still "
              "matches; if load_test_fields' stride or the h5 file changed since the "
              "pilot ran, this comparison is not meaningful.", file=sys.stderr)

    n = min(args.n_fields, p_hr_mm.shape[0], len(old_bpd))
    transform = PrecipTransform(PrecipConfig())
    engine = ConditionalCNFLikelihood(
        velocity,
        LikelihoodConfig(method="rk4", exact_trace=False,
                          hutchinson_probes=args.probes,
                          n_steps_fixed=args.n_steps, use_float64=True),
    )

    print(f"\n{'field':>5} {'old_bpd':>10} {'new_bpd_total':>14} {'new_bpd_interior':>17} "
          f"{'abs_diff':>10} {'nfe':>8}")
    print("-" * 72)
    diffs = []
    for i in range(n):
        p_field = p_hr_mm[i:i + 1].to(args.device)
        c_field = cond_u[i:i + 1].to(args.device)
        dq_gen = torch_generator(field_id=i, condition="sanity_check", replicate=0,
                                  purpose="dequant")
        hu_gen = torch_generator(field_id=i, condition="sanity_check", replicate=0,
                                  purpose="hutchinson")
        out = bits_per_dim(engine, transform, p_field, c_field, dq_gen, hu_gen,
                            d_interior=info.d_interior)
        new_total = out["bpd_total"].item()
        new_interior = out["bits_per_interior_dim"].item()
        diff = abs(new_total - old_bpd[i])
        diffs.append(diff)
        print(f"{i:>5} {old_bpd[i]:>10.4f} {new_total:>14.4f} {new_interior:>17.4f} "
              f"{diff:>10.4f} {out['n_function_evals']:>8}")

    diffs = np.array(diffs)
    print("-" * 72)
    print(f"mean abs diff: {diffs.mean():.4f}   max abs diff: {diffs.max():.4f}")
    print("(E0a's own numerical noise floor at this config was 0.0030 bits/dim per field)")
    if diffs.max() < 0.02:
        print("PASS: differences are consistent with sampling noise, not a bug in the port.")
    elif diffs.max() < 0.1:
        print("BORDERLINE: larger than pure noise would suggest, worth a closer look before "
              "trusting the port for anything downstream.")
    else:
        print("FAIL: differences are far larger than the numerical noise floor. This points "
              "to a real bug in the port (wrong transform config, wrong model wiring, or "
              "wrong field ordering), not sampling variation. Do not proceed to E0b until "
              "this is understood.")


if __name__ == "__main__":
    main()
