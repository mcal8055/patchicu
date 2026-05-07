# Analysis utilities

Post-hoc analysis tools for PatchICU. These are **not** part of YAIB's training pipeline — they consume YAIB outputs (model checkpoints + held-out predictions) and produce additional metrics.

## `mc_calibration.py`

Monte Carlo dropout + isotonic calibration for binary sepsis predictions.

### What it does

1. **MC dropout inference** (`mc_dropout_predict`): runs N stochastic forward passes with Dropout layers selectively enabled, leaving BatchNorm and LayerNorm in eval mode. Returns mean and variance over the N passes via streaming Welford accumulation, so memory is O(prediction-tensor-size) regardless of N.

2. **Isotonic calibration** (`calibrate_and_score`): fits scikit-learn's `IsotonicRegression` on validation MC means and applies the calibrator to test MC means. Reports pre and post calibration AUROC, AUPRC, Brier score, and Expected Calibration Error.

3. **No new dependencies**: torch, scikit-learn, numpy — all present in the YAIB environment.

### Library usage

```python
from analysis.mc_calibration import (
    mc_dropout_predict,
    calibrate_and_score,
    expected_calibration_error,
)

# MC dropout on a trained model
out = mc_dropout_predict(model, batch_x, n_samples=50)
# out['mean']      : [..., num_classes] MC-mean softmax probabilities
#                    (use this for calibration / point-estimate scoring)
# out['epistemic'] : [...] mutual information / BALD score — the formally
#                    correct epistemic uncertainty for classification
# out['aleatoric'] : [...] expected per-sample entropy
# out['total']     : [...] H(mean) — total predictive uncertainty
# out['var']       : [..., num_classes] variance of softmax (diagnostic only;
#                    NOT equivalent to epistemic for classification)

# Calibrate using validation predictions
results = calibrate_and_score(val_probs, val_labels, test_probs, test_labels)
print(results['post_calibration']['ece'])  # post-calibration ECE
```

### CLI usage (consumes precomputed prediction arrays)

```bash
python analysis/mc_calibration.py \
    --val-probs val_probs.npy \
    --val-labels val_labels.npy \
    --test-probs test_probs.npy \
    --test-labels test_labels.npy \
    --output mc_calibration_results.json
```

| File | Shape | Description |
|---|---|---|
| `val_probs.npy` | any (e.g. `[N, time]`) | MC-mean positive-class probabilities |
| `val_labels.npy` | matching shape | Binary labels in {0, 1}; negative values (e.g. -100) treated as masked |
| `test_probs.npy` | any | |
| `test_labels.npy` | matching shape | |

Arrays are flattened internally; any per-timestep / per-batch organization is fine.

### Recipe: extracting MC-mean prediction arrays from a YAIB run

YAIB does not dump per-prediction arrays to disk by default. Two paths to obtain the inputs above:

**Path A — modify YAIB's eval to save predictions.** Add a hook in `icu_benchmarks/models/wrappers.py` (in the `DLPredictionWrapper.test_step` or equivalent) to save predictions per batch. This requires editing YAIB's source.

**Path B — standalone inference script.** Load each fold's `model.ckpt` and rebuild the DataLoader using the parameters in that fold's `train_config.gin`, then iterate with `mc_dropout_predict`. Sketch:

```python
import gin
import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis.mc_calibration import mc_dropout_predict
from icu_benchmarks.models import PatchTST  # or your trained model class
# Plus YAIB's data-loader machinery (the exact API depends on YAIB version):
# from icu_benchmarks.data.split_process_data import preprocess_data
# from icu_benchmarks.data.loader import PredictionPolarsDataset

# 1. Parse the resolved gin config from this fold (gives preprocessing parameters)
gin.parse_config_file('yaib_logs/.../fold_0/train_config.gin')

# 2. Load model from checkpoint
model = PatchTST.load_from_checkpoint('yaib_logs/.../fold_0/model.ckpt').cuda()

# 3. Build val_loader and test_loader for this fold (consult YAIB for the
#    canonical data API in your version; the cohort parquet files live in
#    the dir passed via --data-dir during the original training run).

# 4. Iterate, collecting MC-mean probabilities for the positive class:
def collect(loader):
    means, labels = [], []
    for batch in loader:
        x = batch[0].cuda(non_blocking=True)
        y = batch[1]
        out = mc_dropout_predict(model, x, n_samples=50)
        means.append(out['mean'][..., 1].cpu().numpy())  # positive class
        labels.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))
    return np.concatenate(means), np.concatenate(labels)

val_probs, val_labels = collect(val_loader)
test_probs, test_labels = collect(test_loader)

np.save('val_probs.npy', val_probs)
np.save('val_labels.npy', val_labels)
np.save('test_probs.npy', test_probs)
np.save('test_labels.npy', test_labels)
```

Wiring up YAIB's DataLoader for an arbitrary fold is the brittle bit and depends on the YAIB version. A future release may include a fully-packaged version of this; for now, the algorithms in `mc_calibration.py` are stable and reusable from any inference path that gives you these four arrays.

### Avoiding earlier pitfalls

- **Crashes from N>10**: a previous implementation in this project stacked all N forward passes in a list before averaging — memory grew linearly in N. The streaming aggregation in `mc_dropout_predict` is constant memory in N; N=100 is feasible.
- **Slow inference**: `torch.inference_mode()` (used internally) is faster than `torch.no_grad()`. Combined with streaming aggregation, MC dropout becomes practical at N=50–100.
- **Dependency hell on CHPC**: this module uses only torch, scikit-learn, and numpy — already in the YAIB `requirements.txt`.
- **Don't run MC dropout under `execute_repeated_cv.reproducible = True`**: deterministic mode disables stochastic dropout. Run MC inference as a separate post-eval step, not inside the YAIB training loop.

### References

- Gal & Ghahramani (2016). *Dropout as a Bayesian approximation: Representing model uncertainty in deep learning.* ICML 2016.
- Niculescu-Mizil & Caruana (2005). *Predicting good probabilities with supervised learning.* ICML 2005.
