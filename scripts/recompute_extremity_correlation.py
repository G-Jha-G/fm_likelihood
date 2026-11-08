#!/usr/bin/env python3
"""
recompute_extremity_correlation.py
=====================================

E0c found that rk4-100 (the setting behind every BPD number in the
parent manuscript's Section 4.7, including rho = 0.585) has a bias that
varies by field and trends with extremity. This script exists to answer
the resulting question directly rather than leave it as a worry: does
the BPD-vs-extremity correlation survive when computed under a converged
integrator (dopri5, per E0c's recommendation), or does it change
materially?

This recomputes BPD for ALL 147 test fields (not just the 6 from E0b/E0c)
under dopri5, using K=1 (per E0b) and rtol=1e-4 by default (per E0c,
cheapest setting that passed on every field tested, with room to tighten
via --rtol if you want extra margin). Reports Spearman rho against
lve_extremity (already saved in the pilot's likelihood_extremes.npz)
alongside the recomputed rk4-100 value for direct comparison, plus the
ORIGINAL pilot's own reported rho for a three-way comparison: pilot's
original rk4-100, this script's own rk4-100 (sanity check that we're
comparing apples to apples), and this script's dopri5.

USAGE
-----
    python scripts/recompute_extremity_correlation.py
    python scripts/recompute_extremity_correlation.py --n-fields 147  # full set
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
PROBES = 1  # per E0b


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


def load_completed(results_path: Path) -> dict:
    """field_id -> {"rk4_n100": val, "dopri5": val}"""
    done = {}
    if results_path.exists():
        with open(results_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.setdefault(r["field_id"], {})[r["solver"]] = r["bits_per_interior_dim"]
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-fields", type=int, default=147, help="how many of the 147 to run")
    p.add_argument("--rtol", type=float, default=1e-4, help="dopri5 rtol, per E0c")
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "best_fm_v1.pt"))
    p.add_argument("--npz", type=Path, default=REPO_ROOT / "likelihood_extremes.npz",
                   help="pilot's saved output, for lve_extremity and the original rho")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "extremity_recheck_outputs")
    args = p.parse_args()

    if not args.npz.exists():
        print(f"error: {args.npz} not found")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "extremity_recheck_results.jsonl"
    done = load_completed(results_path)

    old = np.load(args.npz)
    extremity = np.asarray(old["lve_extremity"], dtype=np.float64)
    old_bpd = np.asarray(old["lve_bpd"], dtype=np.float64)
    old_rho = _spearman(extremity[: args.n_fields], old_bpd[: args.n_fields])

    print(f"Pilot's ORIGINAL reported rho (rk4-100, stateful generator): "
          f"{_spearman(extremity, old_bpd):.4f} (full 147 fields)")
    print(f"Same statistic restricted to the first {args.n_fields} fields: {old_rho:.4f}\n")

    print(f"Loading model and data on {args.device}...")
    velocity, p_hr_mm, cond_u, info = load_model_and_data(args.device, args.checkpoint)
    n_fields = min(args.n_fields, p_hr_mm.shape[0])
    field_ids = list(range(n_fields))

    transform = PrecipTransform(PrecipConfig())
    solvers = [
        ("rk4_n100", dict(method="rk4", n_steps_fixed=100)),
        ("dopri5", dict(method="dopri5", rtol=args.rtol, atol=args.rtol)),
    ]
    total_units = len(field_ids) * len(solvers)
    remaining_units = total_units - sum(
        1 for fid in field_ids for name, _ in solvers if name in done.get(fid, {})
    )
    print(f"Fields: {n_fields}  Solvers: {[s[0] for s in solvers]}")
    print(f"Total units: {total_units}  Remaining: {remaining_units}\n")

    completed_count = 0
    elapsed_total = 0.0
    with open(results_path, "a") as out_fh:
        for field_id in field_ids:
            p_field = p_hr_mm[field_id:field_id + 1].to(args.device)
            c_field = cond_u[field_id:field_id + 1].to(args.device)
            for solver_name, solver_kwargs in solvers:
                if solver_name in done.get(field_id, {}):
                    continue
                dq_gen = torch_generator(field_id=field_id, condition="extremity_recheck",
                                          replicate=0, purpose="dequant")
                hu_gen = torch_generator(field_id=field_id, condition="extremity_recheck",
                                          replicate=0, purpose="hutchinson")
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
                    "field_id": field_id, "solver": solver_name,
                    "bits_per_interior_dim": out["bits_per_interior_dim"].item(),
                    "n_function_evals": out["n_function_evals"], "elapsed_sec": dt,
                }
                out_fh.write(json.dumps(record) + "\n")
                out_fh.flush()

                avg = elapsed_total / completed_count
                eta_sec = avg * (remaining_units - completed_count)
                if completed_count % 10 == 0 or completed_count == remaining_units:
                    print(f"[{completed_count}/{remaining_units}] field={field_id} "
                          f"{solver_name}  bpd={record['bits_per_interior_dim']:.4f}  "
                          f"{dt:.1f}s  ETA {eta_sec/60:.1f} min", flush=True)

    print(f"\nDone. New units this run: {completed_count}, "
          f"elapsed: {elapsed_total/60:.1f} min")

    # final correlation comparison from everything in the results file
    all_done = load_completed(results_path)
    fids = sorted(all_done.keys())
    rk4_vals, dopri5_vals, ext_vals = [], [], []
    for fid in fids:
        d = all_done[fid]
        if "rk4_n100" in d and "dopri5" in d:
            rk4_vals.append(d["rk4_n100"])
            dopri5_vals.append(d["dopri5"])
            ext_vals.append(extremity[fid])
    rk4_vals, dopri5_vals, ext_vals = map(np.array, (rk4_vals, dopri5_vals, ext_vals))

    print("\n" + "=" * 78)
    print(f"CORRELATION COMPARISON (n = {len(fids)} fields with both solvers complete)")
    print("=" * 78)
    print(f"Pilot's original reported rho (full 147, different code, stateful RNG): "
          f"{_spearman(extremity, old_bpd):.4f}")
    print(f"This script's rk4-100, same fields                                    : "
          f"{_spearman(ext_vals, rk4_vals):.4f}")
    print(f"This script's dopri5 (converged), same fields                          : "
          f"{_spearman(ext_vals, dopri5_vals):.4f}")
    print(f"Mean bpd shift (dopri5 - rk4_n100), should be positive per E0c         : "
          f"{np.mean(dopri5_vals - rk4_vals):+.4f}")
    if len(fids) < n_fields:
        print(f"\n[note: only {len(fids)}/{n_fields} fields complete, rerun the same "
              f"command to resume and fill in the rest before trusting this comparison]")


if __name__ == "__main__":
    main()
