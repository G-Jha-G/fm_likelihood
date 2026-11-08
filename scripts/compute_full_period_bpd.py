#!/usr/bin/env python3
"""
compute_full_period_bpd.py
=============================

The GPU-bound half of E1's dataset: BPD for all 1461 test-period days
(not the pilot's 147-field stride-10 subsample), under production
settings established by E0: dopri5 (rtol=1e-4 by default, per E0c's
"cheapest setting within tolerance"), K=1 Hutchinson probe (per E0b).

Only computes dopri5, not a repeated rk4-100 comparison, unlike the
extremity recheck. That comparison already ran on 147 fields and showed
a stable, reassuring pattern (rho barely moved and in the safe
direction); redoing it across all 1461 would double this job's ~21 hour
cost for no new information the analysis actually needs.

Resumable the same way as every other GPU script in this project:
results append to a JSONL, a killed or disowned run picks up exactly
where it left off on rerun.

At the end, if e1_covariates.csv exists (from compute_e1_covariates.py),
automatically joins on field_id to produce e1_analysis_table.csv, the
single table E1's actual regression work needs. If it doesn't exist yet,
this script still finishes and just skips that step, rerun later once
the covariates are ready and it'll pick the join up without recomputing
any BPD.

USAGE
-----
    python scripts/compute_full_period_bpd.py                  # all 1461
    python scripts/compute_full_period_bpd.py --n-fields 20      # quick test
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


def load_model_and_data(device: str, checkpoint_path: str, n_fields: int | None):
    """Same loading path as every other script in this project, with the
    stride-10 subsample REMOVED: this is the whole point of this script,
    load every day, not every tenth."""
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
        hr = f["precipitation"][:]  # ALL days, no [::10]
    if n_fields is not None:
        hr = hr[:n_fields]
    hr = hr.astype(np.float32)
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
                if line:
                    done.add(json.loads(line)["field_id"])
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-fields", type=int, default=None,
                   help="limit to the first N days, omit for all 1461")
    p.add_argument("--rtol", type=float, default=1e-4, help="dopri5 rtol, per E0c")
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "best_fm_v1.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "e1_bpd_outputs")
    p.add_argument("--covariates", type=Path, default=REPO_ROOT / "e1_covariates.csv")
    p.add_argument("--print-every", type=int, default=20)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "e1_bpd_results.jsonl"
    done = load_completed(results_path)

    print(f"Loading model and data on {args.device}...")
    velocity, p_hr_mm, cond_u, info = load_model_and_data(
        args.device, args.checkpoint, args.n_fields)
    n_total = p_hr_mm.shape[0]
    print(f"{n_total} fields loaded, padded shape {tuple(p_hr_mm.shape[-2:])}")
    print(f"dopri5 rtol={args.rtol}  K={PROBES} (per E0b)  "
          f"already done: {len(done)}  remaining: {n_total - len(done)}\n")

    transform = PrecipTransform(PrecipConfig())
    cfg = LikelihoodConfig(method="dopri5", rtol=args.rtol, atol=args.rtol,
                            exact_trace=False, hutchinson_probes=PROBES, use_float64=True)
    engine = ConditionalCNFLikelihood(velocity, cfg)

    remaining = n_total - len(done)
    completed_count = 0
    elapsed_total = 0.0
    with open(results_path, "a") as out_fh:
        for field_id in range(n_total):
            if field_id in done:
                continue
            p_field = p_hr_mm[field_id:field_id + 1].to(args.device)
            c_field = cond_u[field_id:field_id + 1].to(args.device)
            dq_gen = torch_generator(field_id=field_id, condition="e1_production",
                                      replicate=0, purpose="dequant")
            hu_gen = torch_generator(field_id=field_id, condition="e1_production",
                                      replicate=0, purpose="hutchinson")

            t0 = time.time()
            out = bits_per_dim(engine, transform, p_field, c_field,
                                dq_gen, hu_gen, d_interior=info.d_interior)
            dt = time.time() - t0
            completed_count += 1
            elapsed_total += dt

            record = {
                "field_id": field_id,
                "bpd_total": out["bpd_total"].item(),
                "bits_per_interior_dim": out["bits_per_interior_dim"].item(),
                "log_prob_model": out["log_prob_model"].item(),
                "log_jac": out["log_jac"].item(),
                "n_function_evals": out["n_function_evals"],
                "elapsed_sec": dt,
            }
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()

            avg = elapsed_total / completed_count
            eta_sec = avg * (remaining - completed_count)
            if completed_count % args.print_every == 0 or completed_count == remaining:
                print(f"[{completed_count}/{remaining}] field={field_id}  "
                      f"bpd={record['bits_per_interior_dim']:.4f}  {dt:.1f}s  "
                      f"ETA {eta_sec/60:.1f} min", flush=True)

    print(f"\nDone. New units this run: {completed_count}, "
          f"total elapsed: {elapsed_total/60:.1f} min")

    # final merge with covariates, if available
    all_done = []
    with open(results_path) as fh:
        for line in fh:
            all_done.append(json.loads(line))
    print(f"\n{len(all_done)}/{n_total} fields complete in {results_path}")

    if not args.covariates.exists():
        print(f"\n[note] {args.covariates} not found, skipping the merge. Run "
              f"compute_e1_covariates.py and rerun this script (it will resume with "
              f"nothing left to compute) to produce the joined table.")
        return

    try:
        import pandas as pd
    except ImportError:
        print("\n[note] pandas not available here, skipping the merge. The two CSVs/JSONL "
              "can be joined on field_id separately.")
        return

    bpd_df = pd.DataFrame(all_done)
    cov_df = pd.read_csv(args.covariates)
    merged = cov_df.merge(bpd_df, on="field_id", how="inner")
    out_path = REPO_ROOT / "e1_analysis_table.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(merged)} rows, {len(merged.columns)} columns), "
          f"the joined table for E1's actual analysis.")
    if len(merged) < min(n_total, len(cov_df)):
        print(f"[note] {min(n_total, len(cov_df)) - len(merged)} field_ids didn't match "
              f"between the BPD results and the covariates, worth checking why before "
              f"trusting the row count.")


if __name__ == "__main__":
    main()
