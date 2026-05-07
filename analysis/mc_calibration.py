"""MC dropout + isotonic calibration for PatchICU.

Post-hoc analysis utilities for trained PatchTST checkpoints. Two layers:

1. ``mc_dropout_predict`` — runs N stochastic forward passes with Dropout layers
   selectively enabled (BatchNorm / LayerNorm stay in eval mode). Streaming
   Welford-style accumulation makes memory O(prediction_size), independent of N.

2. ``calibrate_and_score`` — fits scikit-learn's IsotonicRegression on validation
   MC means and applies the calibrator to test MC means. Returns pre and post
   calibration AUROC, AUPRC, Brier score, and Expected Calibration Error.

Dependencies: torch, scikit-learn, numpy. All present in the YAIB env;
no additional packages required.

References:
    - Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"
    - Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities With
      Supervised Learning" (isotonic baseline)

Usage as a library:
    from analysis.mc_calibration import (
        mc_dropout_predict, calibrate_and_score, expected_calibration_error,
    )

Usage as a CLI (consumes precomputed prediction arrays):
    python analysis/mc_calibration.py \\
        --val-probs val_probs.npy \\
        --val-labels val_labels.npy \\
        --test-probs test_probs.npy \\
        --test-labels test_labels.npy \\
        --output mc_calibration_results.json

See analysis/README.md for the recipe to extract MC-mean prediction arrays
from a trained YAIB checkpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# Build the tuple defensively: nn.Dropout1d only exists in PyTorch >= 1.12.
# Older torch installs would raise AttributeError at import time without this.
DROPOUT_CLASSES = tuple(
    cls for cls in (
        getattr(nn, name, None)
        for name in (
            'Dropout',
            'Dropout1d',
            'Dropout2d',
            'Dropout3d',
            'AlphaDropout',
            'FeatureAlphaDropout',
        )
    )
    if cls is not None
)


# ============================================================================
# Core algorithms
# ============================================================================

@torch.inference_mode()
def mc_dropout_predict(model, x, n_samples=50, eps=1e-12):
    """Run N MC-dropout forward passes; return mean + uncertainty decomposition.

    For classification (softmax outputs), uses the entropy / mutual-information
    decomposition standard in the BALD literature:

        total      = H(mean_p)              # entropy of mean predictive
        aleatoric  = E[H(p_n)]              # expected per-sample entropy
        epistemic  = total - aleatoric      # mutual information / BALD score

    Selectively switches Dropout layers to train mode while leaving the rest of
    the model (BatchNorm, LayerNorm, etc.) in eval mode. Streaming Welford-style
    accumulation: memory is O(prediction_size), independent of n_samples.

    For binary or multiclass classification, the BALD score (``epistemic`` key
    below) is the formally correct epistemic uncertainty measure — variance of
    softmax outputs (kept here as ``var`` for diagnostic interest) is a
    regression-style heuristic and not equivalent.

    Args:
        model: PyTorch ``nn.Module``. Will be set to eval mode internally;
            Dropout layers are then individually flipped to train mode.
        x: Input tensor; shape determined by the model.
        n_samples: Number of MC dropout passes. Larger gives tighter
            uncertainty estimates at linear time cost. 50 is a typical default;
            100+ is feasible thanks to streaming aggregation.
        eps: Small constant added inside log to avoid log(0). Default 1e-12.

    Returns:
        Dict with keys:
            'mean'      : [..., num_classes] MC-mean softmax probabilities
                          (use this for calibration / point-estimate scoring).
            'var'       : [..., num_classes] per-class variance of softmax;
                          diagnostic only — NOT the formally correct epistemic
                          measure for classification.
            'total'     : [...] total predictive uncertainty H(mean_p).
            'aleatoric' : [...] expected per-sample entropy E[H(p_n)].
            'epistemic' : [...] mutual information / BALD score
                          (the correct epistemic uncertainty for classification).

    References:
        - Houlsby et al. (2011), "Bayesian Active Learning for Classification
          and Preference Learning" (BALD score).
        - Smith & Gal (2018), "Understanding Measures of Uncertainty for
          Adversarial Example Detection".
        - Depeweg et al. (2018), "Decomposition of Uncertainty in Bayesian
          Deep Learning for Efficient and Risk-sensitive Learning".
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, DROPOUT_CLASSES):
            m.train()

    sum_p = None
    sum_p2 = None
    sum_entropy = None  # accumulates H(p_n) per pass, for aleatoric

    # Detect MPS for periodic empty_cache (MPS allocator caches intermediate
    # activations aggressively and can OOM during long MC loops otherwise).
    on_mps = (
        hasattr(x, 'device') and x.device.type == 'mps'
        and hasattr(torch, 'mps')
    )

    for i in range(n_samples):
        p = torch.softmax(model(x), dim=-1)
        H_per_sample = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)

        if sum_p is None:
            sum_p = torch.zeros_like(p)
            sum_p2 = torch.zeros_like(p)
            sum_entropy = torch.zeros_like(H_per_sample)

        sum_p += p
        sum_p2 += p * p
        sum_entropy += H_per_sample

        if on_mps and (i + 1) % 5 == 0:
            torch.mps.empty_cache()

    mean = sum_p / n_samples
    var = (sum_p2 / n_samples - mean * mean).clamp_min(0)
    aleatoric = sum_entropy / n_samples
    total = -(mean * torch.log(mean.clamp_min(eps))).sum(dim=-1)
    epistemic = (total - aleatoric).clamp_min(0)

    return {
        'mean': mean,
        'var': var,
        'total': total,
        'aleatoric': aleatoric,
        'epistemic': epistemic,
    }


def expected_calibration_error(probs, labels, n_bins=15):
    """Standard ECE: weighted sum of |bin_accuracy - bin_confidence| across bins.

    Args:
        probs: 1-D array of predicted positive-class probabilities in [0, 1].
        labels: 1-D array of binary ground truth in {0, 1}.
        n_bins: Number of equal-width bins (default 15).

    Returns:
        Float ECE; lower is better, 0 is perfectly calibrated.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: probs {probs.shape} vs labels {labels.shape}"
        )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges) - 1, 0, n_bins - 1)

    n = probs.size
    ece = 0.0
    for i in range(n_bins):
        mask = bin_idx == i
        if not mask.any():
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def calibrate_and_score(val_probs, val_labels, test_probs, test_labels, n_bins=15):
    """Fit isotonic regression on val, apply to test, return pre/post metrics.

    Handles per-timestep predictions: arrays of any shape are flattened
    internally. Negative labels (e.g. -100 from YAIB's
    ``cross_entropy.ignore_index``) are treated as masked-out and excluded
    from both fitting and evaluation.

    Args:
        val_probs: Validation positive-class probabilities (any shape).
        val_labels: Validation binary labels (must broadcast to val_probs
            after the class dim is dropped). Negative values treated as masked.
        test_probs: Test positive-class probabilities.
        test_labels: Test binary labels.
        n_bins: Number of bins for ECE computation (default 15).

    Returns:
        Dict with keys: 'n_test_timesteps', 'pre_calibration', 'post_calibration'.
        Each *_calibration block has 'auroc', 'auprc', 'brier', 'ece'.
    """
    val_p = np.asarray(val_probs).reshape(-1)
    val_y = np.asarray(val_labels).reshape(-1)
    val_mask = val_y >= 0

    test_p = np.asarray(test_probs).reshape(-1)
    test_y = np.asarray(test_labels).reshape(-1)
    test_mask = test_y >= 0

    if val_mask.sum() == 0:
        raise ValueError("val: no labeled timesteps (all labels < 0)")
    if test_mask.sum() == 0:
        raise ValueError("test: no labeled timesteps (all labels < 0)")

    val_p_clean = val_p[val_mask]
    val_y_clean = val_y[val_mask]
    test_p_clean = test_p[test_mask]
    test_y_clean = test_y[test_mask]

    # IsotonicRegression API (sklearn >= 1.0):
    #   IsotonicRegression(*, y_min=None, y_max=None, increasing=True,
    #                      out_of_bounds='nan')
    #   .fit(X, y, sample_weight=None) — both 1-D
    #   .transform(T) — T 1-D, returns 1-D
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    calibrator.fit(val_p_clean, val_y_clean)
    test_p_cal = calibrator.transform(test_p_clean)

    return {
        'n_test_timesteps': int(test_mask.sum()),
        'pre_calibration': _score(test_p_clean, test_y_clean, n_bins),
        'post_calibration': _score(test_p_cal, test_y_clean, n_bins),
    }


def _score(probs, labels, n_bins=15):
    return {
        'auroc': float(roc_auc_score(labels, probs)),
        'auprc': float(average_precision_score(labels, probs)),
        'brier': float(brier_score_loss(labels, probs)),
        'ece': expected_calibration_error(probs, labels, n_bins=n_bins),
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        '--val-probs', required=True,
        help='Path to val MC-mean positive-class probabilities (.npy)',
    )
    ap.add_argument(
        '--val-labels', required=True,
        help='Path to val labels (.npy); negative values treated as masked',
    )
    ap.add_argument(
        '--test-probs', required=True,
        help='Path to test MC-mean positive-class probabilities (.npy)',
    )
    ap.add_argument(
        '--test-labels', required=True,
        help='Path to test labels (.npy)',
    )
    ap.add_argument(
        '--output', default='mc_calibration_results.json',
        help='Output JSON path (default: mc_calibration_results.json)',
    )
    ap.add_argument(
        '--ece-bins', type=int, default=15,
        help='Number of bins for ECE (default: 15)',
    )
    args = ap.parse_args()

    val_p = np.load(args.val_probs)
    val_y = np.load(args.val_labels)
    test_p = np.load(args.test_probs)
    test_y = np.load(args.test_labels)

    print("Loaded:")
    print(f"  val_probs  : shape {val_p.shape}, dtype {val_p.dtype}")
    print(f"  val_labels : shape {val_y.shape}, dtype {val_y.dtype}")
    print(f"  test_probs : shape {test_p.shape}, dtype {test_p.dtype}")
    print(f"  test_labels: shape {test_y.shape}, dtype {test_y.dtype}")

    results = calibrate_and_score(val_p, val_y, test_p, test_y, n_bins=args.ece_bins)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.output}\n")
    pre = results['pre_calibration']
    post = results['post_calibration']
    print(
        f"Pre-calibration : AUROC={pre['auroc']:.4f}  AUPRC={pre['auprc']:.4f}  "
        f"Brier={pre['brier']:.4f}  ECE={pre['ece']:.4f}"
    )
    print(
        f"Post-calibration: AUROC={post['auroc']:.4f}  AUPRC={post['auprc']:.4f}  "
        f"Brier={post['brier']:.4f}  ECE={post['ece']:.4f}"
    )


if __name__ == '__main__':
    main()
