"""Generate diagnostic plots for isotonic calibration outputs.

Reads the .npy prediction arrays saved by ``extract_predictions.py``,
re-fits the isotonic calibrator on val, applies it to test, and produces
two figures saved alongside the predictions:

    calibration.png    — reliability diagrams pre vs post calibration
    discrimination.png — ROC and PR curves pre vs post (overlay)

Uncertainty plotting was removed alongside MC dropout (see
project_v02_demo_uncertainty.md). For uncertainty plots, use a deep
ensemble script that loads all 25 nested-CV checkpoints.

Usage:
    python analysis/make_plots.py --pred-dir predictions/fold_0_mps
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def reliability(probs, labels, n_bins=15):
    """Equal-width binning; returns (bin_mean_pred, bin_pos_rate, bin_count)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    bp = np.zeros(n_bins)
    by = np.zeros(n_bins)
    bn = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = idx == i
        if not m.any():
            continue
        bp[i] = probs[m].mean()
        by[i] = labels[m].mean()
        bn[i] = m.sum()
    return bp, by, bn


def plot_calibration(test_p_pre, test_p_post, test_y, out_path, n_bins=15):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (probs, label) in zip(
        axes,
        [(test_p_pre, "Pre-calibration"), (test_p_post, "Post-calibration")],
    ):
        bp, by, bn = reliability(probs, test_y, n_bins=n_bins)
        valid = bn > 0
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='perfect calibration')
        sizes = 30 + (bn[valid] / max(bn.max(), 1)) * 200
        ax.scatter(bp[valid], by[valid], s=sizes, alpha=0.75,
                   edgecolor='k', linewidth=0.5)
        ax.plot(bp[valid], by[valid], '-', alpha=0.4)
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(label)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    fig.suptitle(
        'Reliability diagrams — equal-width bins; '
        'point size ∝ bin count (capped)'
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_discrimination(test_p_pre, test_p_post, test_y, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr_pre, tpr_pre, _ = roc_curve(test_y, test_p_pre)
    fpr_post, tpr_post, _ = roc_curve(test_y, test_p_post)
    auroc_pre = roc_auc_score(test_y, test_p_pre)
    auroc_post = roc_auc_score(test_y, test_p_post)
    axes[0].plot(fpr_pre, tpr_pre, label=f'Pre  (AUROC={auroc_pre:.4f})')
    axes[0].plot(fpr_post, tpr_post, '--',
                 label=f'Post (AUROC={auroc_post:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate')
    axes[0].set_title('ROC')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    pr_pre = precision_recall_curve(test_y, test_p_pre)
    pr_post = precision_recall_curve(test_y, test_p_post)
    auprc_pre = average_precision_score(test_y, test_p_pre)
    auprc_post = average_precision_score(test_y, test_p_post)
    base_rate = test_y.mean()
    axes[1].plot(pr_pre[1], pr_pre[0], label=f'Pre  (AUPRC={auprc_pre:.4f})')
    axes[1].plot(pr_post[1], pr_post[0], '--',
                 label=f'Post (AUPRC={auprc_post:.4f})')
    axes[1].axhline(base_rate, color='k', linestyle=':', alpha=0.5,
                    label=f'base rate = {base_rate:.4f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        'Discrimination — isotonic calibration is monotonic, so AUROC is '
        'unchanged; AUPRC may shift only via tied-prediction handling'
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        '--pred-dir', required=True, type=Path,
        help='Directory containing val/test prediction .npy files (incl. *_mask.npy)',
    )
    ap.add_argument(
        '--out-dir', type=Path, default=None,
        help='Where to save plots (default: same as --pred-dir)',
    )
    ap.add_argument('--ece-bins', type=int, default=15)
    args = ap.parse_args()

    out = args.out_dir or args.pred_dir
    out.mkdir(parents=True, exist_ok=True)

    val_probs = np.load(args.pred_dir / 'val_probs.npy').reshape(-1)
    val_labels = np.load(args.pred_dir / 'val_labels.npy').reshape(-1)
    val_mask = np.load(args.pred_dir / 'val_mask.npy').reshape(-1).astype(bool)
    test_probs = np.load(args.pred_dir / 'test_probs.npy').reshape(-1)
    test_labels = np.load(args.pred_dir / 'test_labels.npy').reshape(-1)
    test_mask = np.load(args.pred_dir / 'test_mask.npy').reshape(-1).astype(bool)

    val_keep = val_mask & (val_labels >= 0)
    test_keep = test_mask & (test_labels >= 0)

    val_p = val_probs[val_keep]
    val_y = val_labels[val_keep]
    test_p = test_probs[test_keep]
    test_y = test_labels[test_keep]

    cal = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    cal.fit(val_p, val_y)
    test_p_post = cal.transform(test_p)

    plot_calibration(test_p, test_p_post, test_y,
                     out / 'calibration.png', n_bins=args.ece_bins)
    print(f'Wrote {out / "calibration.png"}')
    plot_discrimination(test_p, test_p_post, test_y,
                        out / 'discrimination.png')
    print(f'Wrote {out / "discrimination.png"}')


if __name__ == '__main__':
    main()
