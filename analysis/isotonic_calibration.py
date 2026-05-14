"""Isotonic calibration for PatchICU per-timestep predictions.

Fits scikit-learn's IsotonicRegression on validation probabilities and
applies the calibrator to test probabilities. Reports pre- and post-
calibration AUROC, AUPRC, Brier score, and Expected Calibration Error.

REQUIRES the pad_mask emitted by ``extract_predictions.py``. YAIB's
PredictionPolarsDataset coerces labels at padded and unlabeled timesteps
to 0 (matching pad_value=0.0); without the mask, isotonic learns the
contaminated marginal and reported AUC/Brier/ECE silently include
fake-zero negatives. The mask is the same one YAIB applies in
``DLPredictionWrapper.step_fn`` before computing in-loop metrics.

Usage as a library:
    from analysis.isotonic_calibration import (
        calibrate_and_score, expected_calibration_error,
    )

Usage as a CLI:
    python analysis/isotonic_calibration.py \\
        --val-probs val_probs.npy \\
        --val-labels val_labels.npy \\
        --val-mask val_mask.npy \\
        --test-probs test_probs.npy \\
        --test-labels test_labels.npy \\
        --test-mask test_mask.npy \\
        --output isotonic_results.json

References:
    - Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities With
      Supervised Learning" (isotonic baseline).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


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


def _apply_mask(probs, labels, mask):
    """Flatten + filter to real, labeled timesteps.

    Two filters are AND'd:
      - The pad_mask from YAIB's loader (real labeled timesteps).
      - ``labels >= 0`` (belt-and-suspenders for any downstream consumer
        using a -100 sentinel; no-op for YAIB's {0, 1} labels).
    """
    p = np.asarray(probs).reshape(-1)
    y = np.asarray(labels).reshape(-1)
    m = np.asarray(mask).reshape(-1).astype(bool)
    keep = m & (y >= 0)
    if keep.sum() == 0:
        raise ValueError("no real labeled timesteps after applying mask")
    return p[keep], y[keep]


def calibrate_and_score(
    val_probs, val_labels, val_mask,
    test_probs, test_labels, test_mask,
    n_bins=15,
):
    """Fit isotonic regression on val, apply to test, return pre/post metrics.

    Mask is REQUIRED. See module docstring for why.

    Args:
        val_probs: Validation positive-class probabilities (any shape).
        val_labels: Validation binary labels matching val_probs shape.
        val_mask: Bool array matching val_probs shape; True for real labeled
            timesteps. Padded and unlabeled positions are dropped.
        test_probs, test_labels, test_mask: Same for the test split.
        n_bins: Number of bins for ECE computation (default 15).

    Returns:
        Dict with keys 'n_test_timesteps', 'pre_calibration', 'post_calibration'.
        Each *_calibration block has 'auroc', 'auprc', 'brier', 'ece'.
    """
    val_p, val_y = _apply_mask(val_probs, val_labels, val_mask)
    test_p, test_y = _apply_mask(test_probs, test_labels, test_mask)

    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    calibrator.fit(val_p, val_y)
    test_p_cal = calibrator.transform(test_p)

    return {
        'n_test_timesteps': int(test_p.size),
        'pre_calibration': _score(test_p, test_y, n_bins),
        'post_calibration': _score(test_p_cal, test_y, n_bins),
    }


def _score(probs, labels, n_bins=15):
    return {
        'auroc': float(roc_auc_score(labels, probs)),
        'auprc': float(average_precision_score(labels, probs)),
        'brier': float(brier_score_loss(labels, probs)),
        'ece': expected_calibration_error(probs, labels, n_bins=n_bins),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--val-probs', required=True, help='Path to val probs (.npy)')
    ap.add_argument('--val-labels', required=True, help='Path to val labels (.npy)')
    ap.add_argument('--val-mask', required=True, help='Path to val pad_mask (.npy bool)')
    ap.add_argument('--test-probs', required=True, help='Path to test probs (.npy)')
    ap.add_argument('--test-labels', required=True, help='Path to test labels (.npy)')
    ap.add_argument('--test-mask', required=True, help='Path to test pad_mask (.npy bool)')
    ap.add_argument(
        '--output', default='isotonic_results.json',
        help='Output JSON path (default: isotonic_results.json)',
    )
    ap.add_argument(
        '--ece-bins', type=int, default=15,
        help='Number of bins for ECE (default: 15)',
    )
    args = ap.parse_args()

    val_p = np.load(args.val_probs)
    val_y = np.load(args.val_labels)
    val_m = np.load(args.val_mask)
    test_p = np.load(args.test_probs)
    test_y = np.load(args.test_labels)
    test_m = np.load(args.test_mask)

    print("Loaded:")
    print(f"  val_probs  : shape {val_p.shape}, dtype {val_p.dtype}")
    print(f"  val_labels : shape {val_y.shape}, dtype {val_y.dtype}")
    print(f"  val_mask   : shape {val_m.shape}, real_frac {val_m.mean():.4f}")
    print(f"  test_probs : shape {test_p.shape}, dtype {test_p.dtype}")
    print(f"  test_labels: shape {test_y.shape}, dtype {test_y.dtype}")
    print(f"  test_mask  : shape {test_m.shape}, real_frac {test_m.mean():.4f}")

    results = calibrate_and_score(
        val_p, val_y, val_m, test_p, test_y, test_m, n_bins=args.ece_bins,
    )

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
