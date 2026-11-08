# fm_likelihood

Exact conditional likelihood for a flow-matching precipitation downscaler, with
the experiments, analysis scripts and figure code behind the paper

> Jha, S. K., and V. Gupta, 2026: Exact conditional likelihood in flow-matching
> precipitation downscaling: numerical pitfalls and what the density measures.
> Manuscript in preparation.

The package evaluates the exact conditional log-density log p(x | c) of a
trained flow-matching model by integrating the probability-flow ODE with the
instantaneous change-of-variables formula, and reports it as bits per
interior dimension. The scripts reproduce the paper's numerical validation
(E0), the analysis of what the density tracks (E1), the conditioning and
target-structure perturbation experiments (E3, E2), the null baselines, the
multiple-comparison correction and the screening test.

---

## 1. Installation

```bash
conda activate pytorch_env            # or any environment with Python >= 3.9
pip install -r requirements.txt
python -m pytest tests/               # CPU only, 41 tests
```

`requirements.txt` gives minimum versions, which are not individually tested.
The tests pass with Python 3.10 and torch 2.5.1, torchdiffeq 0.2.5, numpy 1.26.4, scipy 1.15.3, pandas 2.3.2,
h5py 3.8.0, matplotlib 3.10.5 and xarray 2025.4.0 (the authors' server
environment as of October 2026). The full environment is listed in `requirements-lock.txt` (`pip list --format=freeze`).

`torchdiffeq` is required for the adaptive `dopri5` solver used for every
reported number. If it is missing the code falls back to fixed-step RK4 and
prints a warning; those results are not comparable with the paper (the
paper shows the fixed-step setting carries a field-dependent bias).

GPU runs on our server used `export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`
before launching Python, to avoid a `libstdc++` version clash with the
system. You may not need it.

## 2. What you need besides this repository

The likelihood is a property of a trained model and of the data it is
evaluated on, so the experiments need three external things.

| Item | What it is | Where it comes from |
|---|---|---|
| MSWX precipitation | Daily 0.1 degree precipitation, India crop (350 x 350 pixels, 5.05 to 39.95 N, 65.05 to 99.95 E) | Public, GloH2O: https://www.gloh2o.org/mswx/ (Beck et al. 2022). The scripts read the 2011 to 2014 test period (1461 days) from an HDF5 file with dataset key `precipitation`. |
| Trained checkpoint | The flow-matching U-Net of Jha et al. (2026) | A separate file in the Zenodo record, `best_fm_v1.pt` (1.1 GB), with its checksum in `best_fm_v1.pt.md5` |
| `train_precip_fm_v1.py` | Model and normalizer definitions (`Config`, `PrecipNormalizer`, `PrecipUNet`, `compute_lr_fullfield`) | The parent model code of Jha et al. (2026), included in this archive with the authors' server paths replaced by `<DATA_ROOT>`. The experiment scripts import it from the project root. |

The scripts find the data through the `MSWX_TEST_H5` environment variable
(default `data/mswx_test_2011-2014.h5`) and the checkpoint through
`--checkpoint` (default `best_fm_v1.pt` in the project root):

```bash
export MSWX_TEST_H5=/path/to/mswx_test_2011-2014.h5
```

The null baselines (section 5) are fitted to the training period (1979 to 2005),
which is a second HDF5 file passed with `--train-h5 /path/to/mswx_train_1979-2005.h5`.
The numbers in the paper come from a run that used it: B1 and B2 were fitted on
1000 days sampled from that file (excluding the 20 selected fields), and B3's
noise scale on the same days (`analysis_logs/baselines_run_train.log`).

The two quantization checks (`check_mswx_quantization.py`,
`check_mswx_quantization_by_magnitude.py`) read the original MSWX NetCDF
instead, passed with `--nc /path/to/MSWX_Pcp_Daily_Ind_HR.nc`.

The analysis-only scripts (everything in section 5 marked "CPU") need none of
the three: they run from the result tables in this repository. Scripts marked
"MSWX file" also read the MSWX HDF5 data (the training-period file as well, for
the baselines) but need no GPU or checkpoint.

## 3. Package layout

```
fm_likelihood/
  seeds.py       deterministic random streams keyed by (field, condition, replicate, purpose)
  transform.py   log1p and affine map to [-1, 1], exact log-Jacobian, uniform dequantization
  mask.py        padding geometry (to a multiple of 16) and the interior (unpadded) mask
  data.py        India crop and the train / validation / test / unseen period bounds
  velocity.py    adapter that turns any network net([x_t, cond], t) into a velocity field
  cnf.py         the likelihood engine: dopri5 or RK4 integration, Hutchinson trace estimator
  bpd.py         bits per interior dimension from a field and its conditioning
scripts/         experiments, analyses, figures (section 5)
tests/           unit tests (closed-form CNF check, Jacobian, masks, seeds, data crop, defaults)
```

### Quick start

```python
from fm_likelihood.bpd import bits_per_dim
from fm_likelihood.cnf import ConditionalCNFLikelihood, LikelihoodConfig
from fm_likelihood.mask import compute_padding
from fm_likelihood.seeds import torch_generator
from fm_likelihood.transform import PrecipConfig, PrecipTransform
from fm_likelihood.velocity import FMVelocity

velocity = FMVelocity(net)           # net(concat([x_t, cond], 1), t * 1000) -> velocity
engine = ConditionalCNFLikelihood(
    velocity,
    LikelihoodConfig(method="dopri5", rtol=1e-4, atol=1e-4,
                     hutchinson_probes=1, use_float64=True))
transform = PrecipTransform(PrecipConfig())
d_interior = compute_padding(350, 350).d_interior          # 122500

# p_hr_mm: target in mm/day, shape (1, 1, 352, 352): the 350 x 350 field after
#          F.pad(..., mode="reflect") with the offsets from compute_padding(350, 350)
# cond_u:  normalized conditioning, padded the same way
dq = torch_generator(field_id=0, condition="demo", replicate=0, purpose="dequant")
hu = torch_generator(field_id=0, condition="demo", replicate=0, purpose="hutchinson")
out = bits_per_dim(engine, transform, p_hr_mm, cond_u, dq, hu, d_interior=d_interior)
print(out["bits_per_interior_dim"])
```

Always pass seeded generators for anything you report. When you compare two
conditions on the same field (a perturbation against its baseline), build both
generators with the **same** `condition` string so the dequantization noise and
the Hutchinson probe are shared and the difference is not sampling noise.

### Settings that matter

These must match the trained model and the paper:

* `PrecipConfig.dequant_bin_mm = 0.0625`: MSWX stores precipitation in steps
  of 1/16 mm, independent of magnitude. `tests/test_defaults.py` fails if the
  default drifts. An earlier assumption of 0.1 mm was wrong; see
  "Dequantization step" in section 5.
* `PrecipConfig.log1p_max = 6.7859`: log1p of the training-set maximum.
* `LikelihoodConfig`: `dopri5`, `rtol = atol = 1e-4`, one Hutchinson probe,
  float64. These are the settings E0b and E0c select.
* BPD is normalized by the **interior** dimension (122,500 pixels), not the
  padded one. Never substitute `bpd_total` for `bits_per_interior_dim`.

## 4. Data layout the scripts expect

Run everything from the project root (`Likelihood_paper/`). Files used by the
analysis scripts:

| File | Content |
|---|---|
| `e1_analysis_table.csv` | One row per test day: `field_id`, BPD, wet fraction, season and the other E1 covariates. Written by `compute_full_period_bpd.py` |
| `e1_covariates.csv` | Output of `compute_e1_covariates.py` (field-level covariates) |
| `e3_selected_fields.csv`, `e3_swap_partners.csv` | The 20 perturbed fields and their swap partners |
| `e3_outputs/e3_results_FINAL_corrected.jsonl` | E3 results, one record per (field, replicate, axis, level). Use this file, not results from the earlier zero-padded blur (see section 5) |
| `e2_outputs/e2_results.jsonl` | E2 (target-structure) results, same record format |

Result files are resumable JSONL: rerunning a job skips records already
written.

### What is in the archive

The archive mirrors the project folder, so every command in section 5 runs from
its root. Result files sit at the paths the scripts use by default. Three of
them differ from the working folder they came from:

* `e0c_outputs/` and `e1_bpd_outputs/` hold the reruns under the corrected
  0.0625 mm dequantization step.
* `e3_outputs/e3_results_FINAL_corrected.jsonl` holds the E3 results with the
  corrected (reflect-padded) blur.

Figures 1, 3 and 5 in the archive were redrawn from the archive's own result
files by the build script, so they match the tables. Figure 2 (10,000-resample
bootstrap) and Figure 4 are copies of the files the paper uses.

The earlier versions (0.1 mm step; zero-padded blur) are kept only for audit in
`superseded/` and should not be used. `analysis_logs/` has the printed output of
the null-baseline, confound, bootstrap and FDR runs. Of the two baseline logs,
`baselines_run_train.log` is the run behind the paper; `baselines_run_reflect.log`
is an earlier run fitted on the other 365 days of the test file (no `--train-h5`),
kept for the record.

`e0a_outputs/`, `e0b_outputs/` and `extremity_recheck_outputs/` were computed
before the dequantization correction, with the 0.1 mm step, and were not
recomputed. E0b's conclusion was verified at both widths (see "Dequantization
step" in section 5, outputs in `spotcheck_outputs/`). E0a and the extremity
recheck are kept for the record and are not used in the paper's tables or
figures.

In the archive copies of the parent model code, the pilot script and the logs,
the authors' server paths are replaced by `<DATA_ROOT>`; set the data and
checkpoint locations for your own system before running them.

Some docstrings refer to an internal project brief ("brief Section 12", and
similar) that holds the original experiment specifications; it is not
distributed, and each script's own docstring states what it does.

The checkpoint `best_fm_v1.pt` (1.1 GB) is a separate file in the Zenodo
record, with its checksum in `best_fm_v1.pt.md5`. Large files that are not part
of this study (the DDPM checkpoint, the NetCDF prediction files, training logs)
are not included.

## 5. Reproducing the paper

All commands run from the project root. "GPU" means they load the checkpoint;
"CPU" means they only read the result tables. Long GPU jobs are best launched
as `python -u scripts/NAME.py ... > logs/NAME.log 2>&1 &`.

### Numerical validation (Section 3a, Table 1, Figure 1)

| Step | Command | Needs |
|---|---|---|
| Probe-count convergence (E0b) | `python scripts/e0b_probe_convergence.py --full` | GPU |
| Summary of E0b | `python scripts/analyze_e0b_probe_convergence.py --results e0b_outputs/e0b_results.jsonl` | CPU |
| Solver convergence (E0c) | `python scripts/e0c_solver_convergence.py` | GPU |
| Summary of E0c | `python scripts/analyze_e0c_solver_convergence.py` | CPU |
| Integrator effect on the extremity correlation | `python scripts/recompute_extremity_correlation.py --n-fields 147` | GPU |
| Agreement with the pilot implementation | `python scripts/sanity_check_vs_pilot.py --n-fields 5` | GPU |
| Season coverage of the E0 field set | `python scripts/e0_season_stratification_check.py --covariates e1_covariates.csv` | CPU |

Figure 1: `python scripts/make_figure1_numerical_validation.py`

### Dequantization step

The quantization of MSWX was measured, not assumed. An initial assumption of
0.1 mm turned out to be wrong, and the correction is not a constant shift in
BPD. The solver comparison (E0c), the full-period analysis (E1) and the
perturbation experiments (E2, E3) were therefore recomputed under 0.0625 mm.
The probe-count experiment (E0b) was checked at both widths on two fields
instead: the standard deviation of BPD across probe draws changed by less than
0.0005 bits per dimension. Under the corrected width the RK4 bias on the worst
field is larger (0.21 against 0.13 bits per dimension at 100 steps), which is
why E0c was recomputed rather than carried over. The scripts behind this are
kept for the record:

| Step | Command | Needs |
|---|---|---|
| Minimum spacing of stored values | `python scripts/check_mswx_quantization.py --nc /path/to/MSWX_Pcp_Daily_Ind_HR.nc` | NetCDF |
| Spacing within each magnitude bin (uniform step, not float16-like) | `python scripts/check_mswx_quantization_by_magnitude.py --nc /path/to/MSWX_Pcp_Daily_Ind_HR.nc` | NetCDF |
| Closed-form 0.678 bit shift against a recompute under both widths | `python scripts/verify_dequant_bin_correction.py` | GPU |
| E0b and E0c robustness to the bin width (two fields) | `python scripts/spotcheck_e0bc_dequant_bin_robustness.py` | GPU |
| Summary of that spot check | `python scripts/analyze_spotcheck_dequant_bin_robustness.py` | CPU |

### What the density tracks (Section 3b, Table 2, Figure 2)

| Step | Command | Needs |
|---|---|---|
| E1 covariates | `python scripts/compute_e1_covariates.py` | MSWX file |
| BPD for all 1461 test days, joined with the covariates into `e1_analysis_table.csv` (about 21 hours, resumable; run after the covariates) | `python -u scripts/compute_full_period_bpd.py` | GPU |
| Variance decomposition | `python scripts/analyze_e1_confound_decomposition.py --table e1_analysis_table.csv` | CPU |
| Block bootstrap | `python scripts/analyze_e1_bootstrap.py` | CPU |
| Matched-pair analysis | `python scripts/analyze_e1_matched_pairs.py` | CPU |
| Table 2 with BH-FDR over six tests | `python scripts/table2_fdr_correction.py --table e1_analysis_table.csv --n-boot 10000 --matched-pairs --workers 4` | CPU |
| Figure 2 | `python scripts/make_figure2_what_density_measures.py --n-boot 10000` | CPU |

### Response to the conditioning (Section 3c, Table 3, Figure 3)

| Step | Command | Needs |
|---|---|---|
| E3 perturbation runs | `python -u scripts/e3_conditioning_sensitivity.py` | GPU |
| Dose-response analysis | `python scripts/analyze_e3_conditioning_sensitivity.py --results e3_outputs/e3_results_FINAL_corrected.jsonl --n-boot 2000` | CPU |
| Blur axis recomputed after a padding fix (see below) | `python -u scripts/e3_blur_padding_fix_recompute.py --device cuda` | GPU |
| Corrected blur analysis | `python scripts/analyze_e3_blur_corrected.py --results e3_outputs/e3_results_FINAL_corrected.jsonl` | CPU |
| Null baselines B1, B2 and B3 (B3 is the naive conditional baseline added in review) | `python scripts/b1_b2_null_baselines.py --results e3_outputs/e3_results_FINAL_corrected.jsonl --fields-csv e3_selected_fields.csv --partners-csv e3_swap_partners.csv --train-h5 /path/to/mswx_train_1979-2005.h5` | MSWX file |
| Wet-fraction confound check | `python scripts/e3_wetfraction_confound_check.py --results e3_outputs/e3_results_FINAL_corrected.jsonl --fields-csv e3_selected_fields.csv` | MSWX file |
| Swap partner re-derivation | `python scripts/rederive_swap_partners_nn.py --mode derive` | MSWX file, then GPU for the rerun |
| Figure 3 | `python scripts/make_figure3_conditioning_sensitivity.py --results e3_outputs/e3_results_FINAL_corrected.jsonl` | CPU |

`python scripts/b1_b2_null_baselines.py --selftest-only` checks the baseline
code on synthetic data without any input files.

**Blur padding fix.** The first E3 blur implementation let `F.conv2d` zero-pad
the extra border a large Gaussian kernel needs. On a uniform test field, which
any correct blur must leave unchanged, it pulled a corner value from 10.0 to
2.56, so roughly a quarter of the domain was contaminated at the largest sigma
tested. `e3_blur_padding_fix_recompute.py` recomputes only the blur levels with
reflect padding; the baseline, magnitude, shift and swap axes were never
affected. All reported blur results use the corrected values.
`e3_results_FINAL_corrected.jsonl` combines the original run with the recompute:
its baseline, magnitude, shift and swap rows are identical to the original
`e3_results.jsonl`, and its 240 blur rows replace the original ones, matching
`e3_blur_fixed_results.jsonl` to within 5e-5 bits per dimension. The bug had
changed individual blur values by up to 0.032 bits per dimension.

### Target structure and screening (Sections 3d and 3e, Tables 4 and 5, Figures 4 and 5)

| Step | Command | Needs |
|---|---|---|
| E2 target-structure runs (blur and block-shuffle of the target, same 20 fields as E3) | `python -u scripts/e2_structural_sensitivity.py` | GPU |
| Figure 4 | `python scripts/make_figure4_structural_sensitivity.py` | CPU |
| Screening test | `python scripts/screening_test.py --table e1_analysis_table.csv --results e3_outputs/e3_results_FINAL_corrected.jsonl --results e2_outputs/e2_results.jsonl` | CPU |
| Figure 5 | `python scripts/make_figure5_screening.py --csv screening_test_results.csv` | CPU |

The screening score is the BPD residual after removing a cubic in wet fraction
plus season, fitted on the reference real days (out of fold in five contiguous
time blocks). It reports one-sided and two-sided AUC and the true-positive
rate at a 5% false-positive rate, with a bootstrap over fields.

### Bootstrap and estimator validation (Appendix B)

| Step | Command | Needs |
|---|---|---|
| Bootstrap coverage and power | `python scripts/bootstrap_coverage_power.py` | CPU |
| Bias of the rank-based partial-correlation estimator | `python scripts/estimator_bias_check.py` | CPU |

## 6. Reproducibility notes

* **Randomness.** Every stochastic call (dequantization noise, Hutchinson
  probes, surrogates) draws from a stream keyed by
  `(field_id, condition, replicate, purpose)` through `seeds.py`, so a rerun
  gives the same numbers on the same hardware and software.
* **Hardware.** Results were computed in float64 on NVIDIA RTX A5000 GPUs.
  Different GPU or library versions can change the adaptive solver's step
  sequence and shift individual BPD values by amounts of the order of the
  solver tolerance (quantified in E0c), not by amounts that change any
  conclusion in the paper.
* **Runtime.** One likelihood evaluation takes roughly 1 minute on one GPU
  at the reported settings (about 300 function evaluations), which is why
  the GPU experiments are launched as resumable background jobs.

## 7. License

MIT, see `LICENSE`. The MSWX data and the model checkpoint are
distributed under their own terms; check those before redistributing them.

## 8. Acknowledgments

We thank the creators of the MSWX dataset, and the School of Civil and
Environmental Engineering, IIT Mandi, for computing facilities. This work was
supported by the Indian Space Research Organisation (ISRO) through the
RESPOND program.

## 9. Contact

Vivek Gupta (corresponding author), vivekgupta@iitmandi.ac.in.
Shailesh Kumar Jha.
