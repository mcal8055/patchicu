# Reproducing PatchICU Results

Setup and reproduction instructions for the 1-hour HiRID PatchTST results reported in [`REPORT.pdf`](REPORT.pdf). HiRID requires PhysioNet credentialed access and is not redistributed by this repository.

## Prerequisites

### Data access
- **HiRID v1.1.1** — Requires PhysioNet credentialed access and a signed Data Use Agreement. Apply at [physionet.org/content/hirid/1.1.1/](https://physionet.org/content/hirid/1.1.1/). Approval typically takes a few days.
- (Optional, for cross-site validation) **MIMIC-IV v3.1** and **eICU-CRD v2.0** require separate PhysioNet access.

### System
- **Python** 3.10 (matches YAIB upstream)
- **R** 4.2.2 — `ricu` and YAIB-cohorts depend on this version.
- **gcc** 13.1.0 or compatible — required by Arrow and some R packages.
- **GPU** — Sweeps in REPORT.pdf were run on the University of Utah CHPC NVIDIA **GH200** nodes (Grace Hopper, 96GB HBM3e). PatchTST's memory footprint at the configurations in `PatchTST_1hr_tuned.gin` is modest, so smaller cards likely work, but this has not been verified.

## Environment setup

### Python
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` extends upstream YAIB's pin list with `transformers==5.5.3` for the HuggingFace PatchTST port. The `recipies` library is pinned via PyPI as `recipies==1.0`.

### R (cohort generation only)
```bash
git clone https://github.com/rvandewater/YAIB-cohorts.git
cd YAIB-cohorts
git checkout 74ac699
git apply /path/to/patchicu/cohort-modifications.patch
```

(Adjust the patch path to wherever you cloned the patchicu repository.)

Then in R:
```r
renv::restore()
```

Set the `RICU_DATA_PATH` environment variable to the directory containing the HiRID files extracted from the PhysioNet release.

## Cohort regeneration

After applying the patch:

```bash
cd YAIB-cohorts/R
Rscript base_cohort.R --src hirid
Rscript sepsis.R --src hirid
```

Produces the standard HiRID-ICU-Benchmark sepsis cohort:
- `data/base/hirid/` — base cohort parquet files
- `data/sepsis/hirid/` — sepsis-task labeled cohort

Final cohort sizes (matching REPORT.pdf):
- HiRID raw: 33,905 stays
- Base cohort: 32,279 stays
- Sepsis task: **29,642 stays** (stay-level prevalence 6.25%, per-timestep prevalence 2.33%)

## Training

PatchTST follows YAIB's standard training and evaluation interface (the `icu-benchmarks` CLI, registered via `setup.py`). The model is registered via `configs/prediction_models/PatchTST.gin` (the search space used for hyperparameter sweeps). `PatchTST_1hr_tuned.gin` contains the frozen 30-trial Optuna best hyperparameters that produced REPORT.pdf's headline result (AUROC 0.939 / AUPRC 0.292, post-dropout-integration retrain); use this file to reproduce that result directly without re-running the sweep. To re-run the full 30-trial sweep, use `PatchTST.gin` together with the example sweep config in `experiments/`.

For canonical CLI invocations, see [YAIB's main documentation](https://github.com/rvandewater/YAIB#readme). The exact sweep configurations used to produce REPORT.pdf are in `experiments/` (institutional paths scrubbed; adjust for your environment).

### Hardware notes

- Sweeps were distributed across multiple CHPC jobs and not tracked as a single end-to-end wandb run, so wallclock per-trial timing is not directly cited here. Per-job durations can be derived from CHPC `.out` logs if you re-run with similar partition allocation.
- For SLURM-based parallelism, see the example sweep agents in `experiments/chpc_*.sh`.

## Known issues and gotchas

1. **`OMP_NUM_THREADS=1`** — When PatchTST is trained in the same process as a LightGBM baseline (multi-model evaluation runs), set this environment variable to avoid an OpenMP/PyTorch deadlock. Single-model PatchTST training does not require it.
2. **Cross-site prevalence shifts** — MIMIC-IV and eICU produce different sepsis prevalences than HiRID due to case-mix and label-phenotyping differences. Cross-cohort performance numbers are not directly comparable; see REPORT.pdf §5 for discussion.
3. **2-minute resolution variant not included** — The native 2-minute HiRID variant is not in this release. The exploratory 2-minute cohort had a 3.9× prevalence shift relative to the standard 1-hour cohort (46% of stays dropped via undocumented filtering); a literature-comparable rebuild is planned future work.
4. **R/4.2.2 + gcc/13.1.0 modules required** — Older R versions miss `ricu` features; older gcc versions fail to compile some R packages with format-security errors.

## Post-hoc analysis

Optional analysis utilities live in [`analysis/`](analysis/). The pipeline is `extract_predictions.py` (per-fold deterministic inference, saves probs/labels/pad_mask) → `isotonic_calibration.py` (per-fold isotonic calibration via scikit-learn's `IsotonicRegression`, with the pad_mask applied to match YAIB's in-loop scoring) → `aggregate_results.py` (mean/std across the 25 nested-CV folds). No dependencies beyond `requirements.txt`. See [`analysis/README.md`](analysis/README.md) for the full pipeline, CLI flags, and why the pad_mask is required.

## Citing reproductions

See [`README.md`](README.md) for the full citation list (YAIB, HiRID, PatchTST). Reproductions of these specific results should additionally credit this repository (Josh McAlister, u1561737@utah.edu).
