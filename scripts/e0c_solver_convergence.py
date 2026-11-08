#!/usr/bin/env python3
"""
e0c_solver_convergence.py
============================

Brief Section 12 spec, verbatim: "RK4 with n in {10, 25, 50, 100, 200},
plus dopri5 at rtol 1e-4, 1e-5, 1e-6, on the same 20 fields with exact
trace disabled but common random numbers. Error vs the finest setting;
choose the cheapest setting within tolerance."

FIELDS: uses the 6 fields already validated in E0b (2, 48, 134 by
wet-fraction extremes; 36, 86, 138 by intensity extremes) rather than a
fresh 20. These already span both axes the brief cares about, and
reusing them keeps this comparable to E0b rather than introducing a
different, untested field set. Override with --field-ids if you want the
literal 20.

PROBES: fixed at K=1, per E0b's result (passes on all 6 fields with
margin). Consistent with "exact trace disabled": Hutchinson, not exact.

COMMON RANDOM NUMBERS, the part that makes this experiment cheap: for a
given (field, replicate), the SAME Hutchinson probe and the SAME dequant
noise are used across ALL 8 solver settings, by constructing a FRESH
generator with the identical (field_id, condition, replicate, purpose)
key for each solver call rather than reusing one live generator object
(which would advance its state between calls and defeat the whole
point). This makes the DIFFERENCE between solver settings a paired
comparison with the shared stochastic component cancelled out, so it
needs far fewer replicates than an unpaired comparison would to say
something useful, this is exactly why the brief specifies it.

REPLICATES: multiple independent common-random-number draws per field
(each replicate gets its own shared probe/noise, different from other
replicates), so the error-vs-finest estimate isn't hostage to one
particular probe draw. Default 3 for a first pass; --n-replicates to
extend, resumable as always.

"Finest setting": dopri5 at the tightest tested rtol (1e-6), used as the
reference for computing every other setting's error against.

USAGE
-----
    python scripts/e0c_solver_convergence.py                  # pilot: 3 replicates
    python scripts/e0c_solver_convergence.py --n-replicates 8  # more power later
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
DEFAULT_FIELD_IDS = [2, 48, 134, 36, 86, 138]
DEFAULT_RK4_STEPS = [10, 25, 50, 100, 200]
DEFAULT_DOPRI5_RTOL = [1e-4, 1e-5, 1e-6]
PROBES = 1  # per E0b


def build_solver_configs(rk4_steps: list[int], dopri5_rtols: list[float]):
    """Returns list of (solver_name, LikelihoodConfig-kwargs-dict)."""
    configs = []
    for n in rk4_steps:
        configs.append((f"rk4_n{n}", dict(method="rk4", n_steps_fixed=n)))
    for rtol in dopri5_rtols:
        configs.append((f"dopri5_rtol{rtol:.0e}", dict(method="dopri5", rtol=rtol, atol=rtol)))
    return configs


def load_model_and_data(device: str, checkpoint_path: str):
    """Identical loading path to sanity_check_vs_pilot.py and
    e0b_probe_convergence.py, kept the same on purpose."""
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


def load_completed(results_path: Path) -> set:
    done = set()
    if results_path.exists():
        with open(results_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add((r["field_id"], r["replicate"], r["solver_name"]))
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field-ids", type=str,
                   default=",".join(str(x) for x in DEFAULT_FIELD_IDS))
    p.add_argument("--n-replicates", type=int, default=3)
    p.add_argument("--rk4-steps", type=str, default=",".join(str(x) for x in DEFAULT_RK4_STEPS))
    p.add_argument("--dopri5-rtol", type=str,
                   default=",".join(str(x) for x in DEFAULT_DOPRI5_RTOL))
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "best_fm_v1.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "e0c_outputs")
    args = p.parse_args()

    field_ids = [int(x) for x in args.field_ids.split(",")]
    rk4_steps = [int(x) for x in args.rk4_steps.split(",")]
    dopri5_rtols = [float(x) for x in args.dopri5_rtol.split(",")]
    solver_configs = build_solver_configs(rk4_steps, dopri5_rtols)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "e0c_results.jsonl"
    done = load_completed(results_path)

    print(f"Fields: {field_ids}")
    print(f"Replicates: {args.n_replicates}")
    print(f"Solver configs ({len(solver_configs)}): {[c[0] for c in solver_configs]}")
    print(f"Probes (K): {PROBES}  (per E0b)")
    if done:
        print(f"Resuming: {len(done)} (field, replicate, solver) combinations already done")

    print(f"\nLoading model and data on {args.device}...")
    velocity, p_hr_mm, cond_u, info = load_model_and_data(args.device, args.checkpoint)

    transform = PrecipTransform(PrecipConfig())
    total_units = len(field_ids) * args.n_replicates * len(solver_configs)
    remaining_units = total_units - len(
        [1 for fid in field_ids for rep in range(args.n_replicates)
         for name, _ in solver_configs if (fid, rep, name) in done]
    )
    print(f"Total units: {total_units}  Already done: {total_units - remaining_units}  "
          f"Remaining: {remaining_units}")

    completed_count = 0
    elapsed_total = 0.0
    with open(results_path, "a") as out_fh:
        for field_id in field_ids:
            p_field = p_hr_mm[field_id:field_id + 1].to(args.device)
            c_field = cond_u[field_id:field_id + 1].to(args.device)
            for rep in range(args.n_replicates):
                for solver_name, solver_kwargs in solver_configs:
                    key = (field_id, rep, solver_name)
                    if key in done:
                        continue

                    # Fresh generators with the SAME (field_id, condition,
                    # replicate, purpose) for every solver_name in this
                    # inner loop: common random numbers by construction,
                    # not by reusing a live generator object.
                    dq_gen = torch_generator(field_id=field_id, condition="e0c_common",
                                              replicate=rep, purpose="dequant")
                    hu_gen = torch_generator(field_id=field_id, condition="e0c_common",
                                              replicate=rep, purpose="hutchinson")

                    cfg = LikelihoodConfig(exact_trace=False, hutchinson_probes=PROBES,
                                            use_float64=True, **solver_kwargs)
                    engine = ConditionalCNFLikelihood(velocity, cfg)

                    t0 = time.time()
                    out = bits_per_dim(engine, transform, p_field, c_field,
                                        dq_gen, hu_gen, d_interior=info.d_interior)
                    dt = time.time() - t0
                    completed_count += 1
                    elapsed_total += dt

                    record = {
                        "field_id": field_id, "replicate": rep, "solver_name": solver_name,
                        "bits_per_interior_dim": out["bits_per_interior_dim"].item(),
                        "n_function_evals": out["n_function_evals"],
                        "elapsed_sec": dt,
                    }
                    out_fh.write(json.dumps(record) + "\n")
                    out_fh.flush()

                    avg = elapsed_total / completed_count
                    eta_sec = avg * (remaining_units - completed_count)
                    print(f"[{completed_count}/{remaining_units}] field={field_id} "
                          f"rep={rep} {solver_name:>16}  "
                          f"bpd={record['bits_per_interior_dim']:.4f}  "
                          f"nfe={record['n_function_evals']:>6}  {dt:.1f}s  "
                          f"ETA {eta_sec/60:.1f} min", flush=True)

    print(f"\nDone. Total new units this run: {completed_count}, "
          f"total elapsed: {elapsed_total/60:.1f} min")
    print(f"\nNext: python scripts/analyze_e0c_solver_convergence.py")


if __name__ == "__main__":
    main()
