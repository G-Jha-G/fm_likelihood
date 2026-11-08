#!/usr/bin/env python3
"""
e0b_probe_convergence.py
==========================

Brief Section 12 spec, verbatim: "K in {1, 2, 4, 8, 16, 32, 64}, 20 fields
spanning the wetness range, 5 seeds each. SD of BPD vs K; choose the
smallest K with SD < 0.02 bits/dim."

This is the first script in the project with a genuinely nontrivial
compute cost: 20 fields x 7 K values x 5 seeds = 700 log_prob calls, and
at K=64 a single call costs nfe = n_steps*4*(1+64), about 7.2x the K=8
call the sanity check just ran. Two things follow from that:

1. RESUMABLE. Results are appended to a JSONL file as each (field, K,
   seed) completes. Killed, disowned, or interrupted midway, rerunning
   the same command skips everything already done and picks up where it
   left off, per the brief's own "Job logging" convention (Section 11).

2. TIMED PILOT FIRST. Default invocation runs a small subset (3 fields,
   2 seeds, K in {1, 8, 64} only) and reports both elapsed time and a
   projected time for the full 20 x 5 x 7 sweep, before you commit to it.
   Run with --full once you've seen that projection and decided it's
   worth the wall clock.

FIELD SELECTION: "spanning the wetness range" is implemented as the wet
fraction (fraction of interior pixels above the wet threshold) computed
for all 147 test fields, then N_FIELDS indices evenly spaced across the
SORTED wet-fraction order, so the selected fields genuinely span dry to
wet rather than clustering near the middle of the distribution.

USAGE
-----
    python scripts/e0b_probe_convergence.py                 # timed pilot
    python scripts/e0b_probe_convergence.py --full            # full sweep
    python scripts/e0b_probe_convergence.py --full            # rerun: resumes
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
from fm_likelihood.mask import compute_padding, to_interior
from fm_likelihood.seeds import torch_generator
from fm_likelihood.transform import PrecipConfig, PrecipTransform
from fm_likelihood.velocity import FMVelocity

MSWX_TEST_H5 = os.environ.get("MSWX_TEST_H5", "data/mswx_test_2011-2014.h5")
WET_THRESHOLD_MM = 1.0


def load_model_and_data(device: str, checkpoint_path: str):
    """Same loading path as sanity_check_vs_pilot.py, kept identical on
    purpose so every script scores the same 147 fields in the same order."""
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


def select_wetness_spanning_fields(p_hr_mm: torch.Tensor, info, n_fields: int) -> list[int]:
    """Wet fraction per field on the INTERIOR only (padding is duplicated
    data, would bias the wet-fraction estimate), then n_fields indices
    evenly spaced through the sorted order."""
    interior = to_interior(p_hr_mm, info)  # (N, 1, H, W)
    wet_frac = (interior > WET_THRESHOLD_MM).float().mean(dim=(1, 2, 3))
    order = torch.argsort(wet_frac)
    n_total = len(order)
    pick_positions = torch.linspace(0, n_total - 1, n_fields).round().long()
    pick_positions = torch.unique(pick_positions)
    selected = order[pick_positions].tolist()
    return selected


def load_completed(results_path: Path) -> set:
    done = set()
    if results_path.exists():
        with open(results_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add((r["field_id"], r["K"], r["seed"]))
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true",
                   help="run the brief's exact spec: 20 fields, K in {1,2,4,8,16,32,64}, "
                        "5 seeds. Without this flag, runs a small timed pilot instead.")
    p.add_argument("--n-fields", type=int, default=None)
    p.add_argument("--field-ids", type=str, default=None,
                   help="comma-separated explicit field indices, e.g. '2,48,134'. "
                        "Overrides --n-fields and the automatic wetness-spanning "
                        "selection. Use this once you have a field set you care "
                        "about, since changing --n-fields alone reselects DIFFERENT "
                        "fields each time (the evenly-spaced wetness positions shift "
                        "with n_fields), which silently breaks resumability: two runs "
                        "with different --n-fields are not guaranteed to share any "
                        "field in common, even though both claim to 'span the "
                        "wetness range'.")
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--probe-list", type=str, default=None,
                   help="comma-separated K values, e.g. '1,8,64'")
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "best_fm_v1.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "e0b_outputs")
    args = p.parse_args()

    if args.full:
        n_fields = args.n_fields or 20
        n_seeds = args.n_seeds or 5
        probe_list = [int(x) for x in (args.probe_list or "1,2,4,8,16,32,64").split(",")]
        mode = "FULL SWEEP (brief spec)"
    else:
        n_fields = args.n_fields or 3
        n_seeds = args.n_seeds or 2
        probe_list = [int(x) for x in (args.probe_list or "1,8,64").split(",")]
        mode = "TIMED PILOT (pass --full for the real sweep once you've seen the ETA)"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "e0b_results.jsonl"
    done = load_completed(results_path)
    print(f"Mode: {mode}")
    print(f"n_fields={n_fields}  n_seeds={n_seeds}  probes={probe_list}  n_steps={args.n_steps}")
    if done:
        print(f"Resuming: {len(done)} (field, K, seed) combinations already completed "
              f"in {results_path}")

    print(f"Loading model and data on {args.device}...")
    velocity, p_hr_mm, cond_u, info = load_model_and_data(args.device, args.checkpoint)
    if args.field_ids:
        field_ids = [int(x) for x in args.field_ids.split(",")]
        print(f"Using explicit field list ({len(field_ids)} fields): {field_ids}")
    else:
        field_ids = select_wetness_spanning_fields(p_hr_mm, info, n_fields)
        print(f"Auto-selected {len(field_ids)} fields spanning the wetness range: "
              f"{field_ids}")

    transform = PrecipTransform(PrecipConfig())
    total_units = len(field_ids) * len(probe_list) * n_seeds
    remaining_units = total_units - len(
        [1 for fid in field_ids for K in probe_list for s in range(n_seeds)
         if (fid, K, s) in done]
    )
    print(f"Total units: {total_units}  Already done: {total_units - remaining_units}  "
          f"Remaining: {remaining_units}")

    completed_count = 0
    elapsed_total = 0.0
    with open(results_path, "a") as out_fh:
        for field_id in field_ids:
            p_field = p_hr_mm[field_id:field_id + 1].to(args.device)
            c_field = cond_u[field_id:field_id + 1].to(args.device)
            for K in probe_list:
                engine = ConditionalCNFLikelihood(
                    velocity,
                    LikelihoodConfig(method="rk4", exact_trace=False,
                                      hutchinson_probes=K, n_steps_fixed=args.n_steps,
                                      use_float64=True),
                )
                for seed in range(n_seeds):
                    key = (field_id, K, seed)
                    if key in done:
                        continue
                    t0 = time.time()
                    dq_gen = torch_generator(field_id=field_id, condition="e0b",
                                              replicate=0, purpose="dequant")
                    hu_gen = torch_generator(field_id=field_id, condition=f"e0b_K{K}",
                                              replicate=seed, purpose="hutchinson")
                    out = bits_per_dim(engine, transform, p_field, c_field,
                                        dq_gen, hu_gen, d_interior=info.d_interior)
                    dt = time.time() - t0
                    completed_count += 1
                    elapsed_total += dt

                    record = {
                        "field_id": field_id, "K": K, "seed": seed,
                        "bpd_total": out["bpd_total"].item(),
                        "bits_per_interior_dim": out["bits_per_interior_dim"].item(),
                        "n_function_evals": out["n_function_evals"],
                        "elapsed_sec": dt,
                    }
                    out_fh.write(json.dumps(record) + "\n")
                    out_fh.flush()

                    avg = elapsed_total / completed_count
                    eta_sec = avg * (remaining_units - completed_count)
                    print(f"[{completed_count}/{remaining_units}] field={field_id} K={K:>2} "
                          f"seed={seed}  bpd={record['bits_per_interior_dim']:.4f}  "
                          f"{dt:.1f}s  ETA {eta_sec/60:.1f} min", flush=True)

    print(f"\nDone. Total new units this run: {completed_count}, "
          f"total elapsed: {elapsed_total/60:.1f} min")
    if not args.full:
        per_unit = elapsed_total / max(completed_count, 1)
        full_units = 20 * 5 * 7
        print(f"\nProjected time for the FULL spec (20 fields x 5 seeds x 7 K values "
              f"= {full_units} units) at this per-unit rate: "
              f"~{per_unit * full_units / 60:.1f} min "
              f"({per_unit * full_units / 3600:.2f} hours)")
        print("Note: this projection assumes uniform per-unit cost, but cost scales with K "
              "(nfe = n_steps*4*(1+K)), so it is a rough average across the pilot's K values, "
              "not exact. Run with --full when ready; it will resume from what's already here.")

    # summarise SD(K) from everything in the results file so far, not just this run
    all_done = []
    with open(results_path) as fh:
        for line in fh:
            all_done.append(json.loads(line))
    if all_done:
        by_k: dict = {}
        for r in all_done:
            by_k.setdefault(r["K"], []).append(r["bits_per_interior_dim"])
        print("\nSD(K) so far (pooled across whatever fields/seeds are in the results file):")
        print(f"{'K':>4} {'n':>5} {'SD':>8}")
        for K in sorted(by_k):
            vals = by_k[K]
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            flag = "  <- meets 0.02 target" if sd < 0.02 else ""
            print(f"{K:>4} {len(vals):>5} {sd:>8.4f}{flag}")
        print("\nNote: this pooled SD mixes between-field and within-field variance; it's a "
              "quick read while the sweep runs. For the brief's actual decision rule (SD of "
              "BPD across seeds, PER FIELD, then how that shrinks with K), rerun the full "
              "spec and analyse e0b_results.jsonl grouped by field_id, not pooled.")


if __name__ == "__main__":
    main()
