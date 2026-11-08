#!/usr/bin/env python3
"""
spotcheck_e0bc_dequant_bin_robustness.py
===========================================

E0b (K=1 sufficient) and E0c (rk4-100 biased by up to 0.13 bits/dim,
dopri5 corrects it) were both computed under the WRONG dequant bin
(0.1 mm, corrected to 0.0625 mm per Q7). The argument that they're
unaffected: both experiments held the dequant generator draw CONSTANT
across everything being compared (E0b: across K and seed, for a fixed
field; E0c: across all 8 solver settings, for a fixed field/replicate,
via common random numbers). Since the log-Jacobian term depends only on
the dequantised value, not on K or the solver, it should be an identical
additive constant across whatever is being compared, and cancel exactly
out of both an SD-across-seeds calculation (E0b) and a bias-vs-reference
calculation (E0c).

This is exactly the shape of argument that turned out WRONG for E1
(the "it's just a constant shift, ranks and variance are unaffected"
claim), so it is not being trusted again without a direct check. Unlike
that mistake, this script does not rely on comparing fresh numbers
against remembered old ones: it recomputes E0b-style and E0c-style
results under BOTH bin widths, in the same run, for the same fields and
settings, so the comparison is self-contained and immune to a
transcription error on either side.

SCOPE, deliberately bounded, not a full rerun
------------------------------------------------
Two fields only: field 2 (driest, smallest E0c bias originally) and
field 138 (the single worst-case field for both wet-fraction-adjacent
selection criteria and the largest RK4 bias found in the original E0c).
These bracket the range where a real effect would be most likely to
show up.

  E0b-style: 5 seeds (not the original 10) x K=1 x rk4-100, both bins.
  E0c-style: 2 replicates (not the original 3) x 4 settings (rk4_n100,
             rk4_n200, dopri5_rtol1e-4, dopri5_rtol1e-6 as reference),
             both bins, common random numbers within each bin.

Expected runtime: on the order of 2 hours (roughly 1/10th the cost of
redoing E0b+E0c in full), resumable if interrupted.

USAGE
-----
    python scripts/spotcheck_e0bc_dequant_bin_robustness.py
"""
from __future__ import annotations

import os
import json
import sys
import time
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
FIELD_IDS = [2, 138]
BINS = {"old": 0.1, "new": 0.0625}
E0B_SEEDS = 5
E0C_REPLICATES = 2
E0C_SOLVERS = [
    ("rk4_n100", dict(method="rk4", n_steps_fixed=100)),
    ("rk4_n200", dict(method="rk4", n_steps_fixed=200)),
    ("dopri5_rtol1e-4", dict(method="dopri5", rtol=1e-4, atol=1e-4)),
    ("dopri5_rtol1e-6", dict(method="dopri5", rtol=1e-6, atol=1e-6)),
]
FINEST = "dopri5_rtol1e-6"


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


def load_completed(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["check"], r["field_id"], r.get("seed"), r.get("replicate"),
                              r.get("solver_name"), r["bin_label"]))
    return done


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = str(REPO_ROOT / "best_fm_v1.pt")
    out_dir = REPO_ROOT / "spotcheck_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "spotcheck_results.jsonl"
    done = load_completed(results_path)

    print(f"Loading model and data on {device}...")
    velocity, p_hr_mm, cond_u, info = load_model_and_data(device, checkpoint)
    transform_cfgs = {label: PrecipTransform(PrecipConfig(dequant_bin_mm=b))
                       for label, b in BINS.items()}

    total_units = (len(FIELD_IDS) * E0B_SEEDS * len(BINS)
                   + len(FIELD_IDS) * E0C_REPLICATES * len(E0C_SOLVERS) * len(BINS))
    print(f"Total units: {total_units}  Already done: {len(done)}\n")

    completed_count, elapsed_total = 0, 0.0
    remaining = total_units - len(done)

    with open(results_path, "a") as out_fh:
        for field_id in FIELD_IDS:
            p_field = p_hr_mm[field_id:field_id + 1].to(device)
            c_field = cond_u[field_id:field_id + 1].to(device)
            engine = ConditionalCNFLikelihood(
                velocity, LikelihoodConfig(method="rk4", n_steps_fixed=100,
                                            exact_trace=False, hutchinson_probes=1,
                                            use_float64=True))
            for bin_label in BINS:
                for seed in range(E0B_SEEDS):
                    key = ("e0b", field_id, seed, None, None, bin_label)
                    if key in done:
                        continue
                    dq_gen = torch_generator(field_id=field_id, condition="e0b_spotcheck",
                                              replicate=0, purpose="dequant")
                    hu_gen = torch_generator(field_id=field_id, condition="e0b_spotcheck",
                                              replicate=seed, purpose="hutchinson")
                    t0 = time.time()
                    out = bits_per_dim(engine, transform_cfgs[bin_label], p_field, c_field,
                                        dq_gen, hu_gen, d_interior=info.d_interior)
                    dt = time.time() - t0
                    completed_count += 1
                    elapsed_total += dt
                    record = {"check": "e0b", "field_id": field_id, "seed": seed,
                              "replicate": None, "solver_name": None, "bin_label": bin_label,
                              "bits_per_interior_dim": out["bits_per_interior_dim"].item()}
                    out_fh.write(json.dumps(record) + "\n")
                    out_fh.flush()
                    eta = (elapsed_total / completed_count) * (remaining - completed_count)
                    print(f"[{completed_count}/{remaining}] e0b field={field_id} "
                          f"bin={bin_label} seed={seed}  {dt:.1f}s  ETA {eta/60:.1f} min",
                          flush=True)

        for field_id in FIELD_IDS:
            p_field = p_hr_mm[field_id:field_id + 1].to(device)
            c_field = cond_u[field_id:field_id + 1].to(device)
            for bin_label in BINS:
                for rep in range(E0C_REPLICATES):
                    for solver_name, solver_kwargs in E0C_SOLVERS:
                        key = ("e0c", field_id, None, rep, solver_name, bin_label)
                        if key in done:
                            continue
                        dq_gen = torch_generator(field_id=field_id, condition="e0c_spotcheck",
                                                  replicate=rep, purpose="dequant")
                        hu_gen = torch_generator(field_id=field_id, condition="e0c_spotcheck",
                                                  replicate=rep, purpose="hutchinson")
                        cfg = LikelihoodConfig(exact_trace=False, hutchinson_probes=1,
                                                use_float64=True, **solver_kwargs)
                        engine = ConditionalCNFLikelihood(velocity, cfg)
                        t0 = time.time()
                        out = bits_per_dim(engine, transform_cfgs[bin_label], p_field, c_field,
                                            dq_gen, hu_gen, d_interior=info.d_interior)
                        dt = time.time() - t0
                        completed_count += 1
                        elapsed_total += dt
                        record = {"check": "e0c", "field_id": field_id, "seed": None,
                                  "replicate": rep, "solver_name": solver_name,
                                  "bin_label": bin_label,
                                  "bits_per_interior_dim": out["bits_per_interior_dim"].item()}
                        out_fh.write(json.dumps(record) + "\n")
                        out_fh.flush()
                        eta = (elapsed_total / completed_count) * (remaining - completed_count)
                        print(f"[{completed_count}/{remaining}] e0c field={field_id} "
                              f"bin={bin_label} rep={rep} {solver_name}  {dt:.1f}s  "
                              f"ETA {eta/60:.1f} min", flush=True)

    print(f"\nDone. New units this run: {completed_count}, "
          f"elapsed: {elapsed_total/60:.1f} min")
    print("\nNext: python scripts/analyze_spotcheck_dequant_bin_robustness.py")


if __name__ == "__main__":
    main()
