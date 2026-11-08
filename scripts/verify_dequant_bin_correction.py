#!/usr/bin/env python3
"""
verify_dequant_bin_correction.py
===================================

Q7 resolved: MSWX's real precision is a uniform 0.0625 mm (1/16 mm),
confirmed flat across every magnitude bin from 0.25 to 512 mm/day, not
the assumed 0.1 mm and not magnitude-dependent (ruled out float16-style
scaling directly).

The bpd formula's normalization term (-D*log(dequant_bin_mm)/(D*log2))
means correcting this is, in theory, a pure closed-form additive shift:
    bpd_corrected = bpd_original + log(0.1/0.0625)/log(2) = bpd_original + 0.678072
identical for every field, which would mean every rank-based and
variance-based result in this study (every correlation, every R^2) is
completely unaffected, only absolute BPD values change.

That argument has one gap this script closes: dequantisation noise
itself was drawn from the WRONG width (U[0, 0.1) instead of U[0, 0.0625)),
which changes the actual values fed into the network, not just a
bookkeeping constant. The theoretical argument assumes this is
negligible. This script checks that directly rather than assuming it:
recompute BPD for the same 6 fields used throughout E0, under BOTH bin
widths, with correctly-redrawn dequantisation noise at each width, and
compare the actual empirical shift against the theoretical 0.678072.

If they agree to within the noise floor E0a already established
(0.0030 bits/dim), the closed-form correction is validated and no
GPU recompute of the full 1461-day dataset is needed, every existing
result can be corrected by simple arithmetic. If they disagree by more
than that, the second-order effect is real and a fuller recompute is
warranted.

USAGE
-----
    python scripts/verify_dequant_bin_correction.py
"""
from __future__ import annotations

import os
import math
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
FIELD_IDS = [2, 48, 134, 36, 86, 138]  # same 6 fields used throughout E0
OLD_BIN, NEW_BIN = 0.1, 0.0625
THEORETICAL_SHIFT = math.log(OLD_BIN / NEW_BIN) / math.log(2)


def load_model_and_data(device: str, checkpoint_path: str):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = str(REPO_ROOT / "best_fm_v1.pt")

    print(f"Theoretical shift (closed-form): {THEORETICAL_SHIFT:.6f} bits/dim")
    print(f"E0a's established noise floor (the tolerance for agreement): 0.0030 bits/dim\n")

    print(f"Loading model and data on {device}...")
    velocity, p_hr_mm, cond_u, info = load_model_and_data(device, checkpoint)

    engine = ConditionalCNFLikelihood(
        velocity, LikelihoodConfig(method="dopri5", rtol=1e-4, atol=1e-4,
                                    exact_trace=False, hutchinson_probes=1, use_float64=True))

    print(f"\n{'field':>6} {'bpd(0.1)':>10} {'bpd(0.0625)':>12} "
          f"{'actual shift':>13} {'theoretical':>12} {'diff':>8}")
    print("-" * 68)
    diffs = []
    for field_id in FIELD_IDS:
        p_field = p_hr_mm[field_id:field_id + 1].to(device)
        c_field = cond_u[field_id:field_id + 1].to(device)

        results = {}
        for bin_width, label in [(OLD_BIN, "old"), (NEW_BIN, "new")]:
            cfg = PrecipConfig(dequant_bin_mm=bin_width)
            tr = PrecipTransform(cfg)
            dq_gen = torch_generator(field_id=field_id, condition="dequant_bin_check",
                                      replicate=0, purpose="dequant")
            hu_gen = torch_generator(field_id=field_id, condition="dequant_bin_check",
                                      replicate=0, purpose="hutchinson")
            out = bits_per_dim(engine, tr, p_field, c_field, dq_gen, hu_gen,
                                d_interior=info.d_interior)
            results[label] = out["bits_per_interior_dim"].item()

        actual_shift = results["new"] - results["old"]
        diff = actual_shift - THEORETICAL_SHIFT
        diffs.append(diff)
        print(f"{field_id:>6} {results['old']:>10.4f} {results['new']:>12.4f} "
              f"{actual_shift:>13.6f} {THEORETICAL_SHIFT:>12.6f} {diff:>8.5f}")

    diffs = np.array(diffs)
    print("-" * 68)
    print(f"mean |diff from theoretical|: {np.abs(diffs).mean():.5f}  "
          f"max |diff|: {np.abs(diffs).max():.5f}")
    print(f"(compare against E0a's noise floor: 0.0030 bits/dim)\n")

    if np.abs(diffs).max() < 0.0030:
        print("PASS: the empirical shift matches the closed-form theoretical shift within "
              "the established noise floor. The second-order effect (redrawn dequant noise "
              "at the correct width) is negligible. Every existing BPD value in this study "
              "can be corrected by simple addition of +0.678072, no GPU recompute needed. "
              "All rank-based and variance-based results (every correlation, every R^2) are "
              "unaffected and require no changes at all.")
    else:
        print("NOT CONFIRMED: the empirical shift differs from theory by more than the noise "
              "floor. The second-order effect is not negligible after all; a full recompute "
              "of the affected results under the corrected bin width is warranted before "
              "trusting the closed-form correction.")


if __name__ == "__main__":
    main()
