# Analysis utilities

Post-hoc analysis tools for PatchICU. These are **not** part of YAIB's training pipeline — they consume YAIB outputs (model checkpoints + held-out predictions) and produce additional metrics.

## Pipeline

```
extract_predictions.py   per-fold inference → val/test_(probs|labels|mask).npy
isotonic_calibration.py  per-fold isotonic + scoring → results.json
aggregate_results.py     mean/std across 25 folds → aggregated_results.json
make_plots.py            optional: reliability + ROC/PR plots from a fold dir
```

Shell wrappers: `run_inference.sh` (one fold), `run_all_folds.sh` (25 folds end-to-end), `run_calibrate.sh` / `run_plots.sh` (re-run downstream on existing predictions).

## Why the mask matters

YAIB's `PredictionPolarsDataset` returns a `(data, labels, pad_mask)` 3-tuple. At padded and unlabeled timesteps the label is coerced to 0 (matching `pad_value=0.0`), and the mask channel carries the real-labeled signal. YAIB's `DLPredictionWrapper.step_fn` applies `torch.masked_select` before computing in-loop metrics; any downstream scorer that doesn't apply the mask silently includes those fake-zero negatives and reports a contaminated AUC/Brier/ECE.

`extract_predictions.py` always saves the mask; `isotonic_calibration.py` and `make_plots.py` require it.

## `extract_predictions.py`

Single deterministic forward pass per fold (no MC dropout — the GP-tuned PatchICU model converged on dropout≈0, so MC sampling produced near-zero epistemic variance; see `project_v02_demo_uncertainty.md`).

```bash
python analysis/extract_predictions.py \
    --fold-dir ../yaib_logs/.../repetition_0/fold_0 \
    --data-dir ../YAIB-cohorts/data/sepsis/hirid \
    --output-dir predictions/fold_0_mps \
    --device mps
```

Outputs (per split, in `--output-dir`):

| File | Shape | Description |
|---|---|---|
| `val_probs.npy` | `[n_stays, time]` | Positive-class softmax probabilities |
| `val_labels.npy` | `[n_stays, time]` | Binary labels (0 at padded positions, per YAIB) |
| `val_mask.npy` | `[n_stays, time]` bool | True for real labeled timesteps |
| `test_probs.npy` | `[n_stays, time]` | |
| `test_labels.npy` | `[n_stays, time]` | |
| `test_mask.npy` | `[n_stays, time]` bool | |

## `isotonic_calibration.py`

Fits scikit-learn's `IsotonicRegression` on the masked val marginal, applies it to the masked test split, reports pre/post AUROC, AUPRC, Brier, ECE.

```bash
python analysis/isotonic_calibration.py \
    --val-probs val_probs.npy \
    --val-labels val_labels.npy \
    --val-mask val_mask.npy \
    --test-probs test_probs.npy \
    --test-labels test_labels.npy \
    --test-mask test_mask.npy \
    --output results.json
```

Library use:

```python
from analysis.isotonic_calibration import calibrate_and_score, expected_calibration_error

results = calibrate_and_score(
    val_probs, val_labels, val_mask,
    test_probs, test_labels, test_mask,
)
print(results['post_calibration']['ece'])
```

## Uncertainty quantification

MC dropout was retired (dropout≈0 in the trained model). For epistemic uncertainty, use a deep ensemble across the 25 nested-CV checkpoints — see `project_v02_demo_uncertainty.md`. No ensemble script lives here yet.

## References

- Niculescu-Mizil & Caruana (2005). *Predicting good probabilities with supervised learning.* ICML 2005.
- Lakshminarayanan et al. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles.* NeurIPS 2017.
