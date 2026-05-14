#!/usr/bin/env bash
# Run isotonic calibration + scoring on a directory of saved predictions.
# Run with: bash analysis/run_calibrate.sh [predictions_dir]
#
# Default: predictions/fold_0_mps (matches default of run_inference.sh)

set -e

DIR="${1:-predictions/fold_0_mps}"

if [ ! -d "$DIR" ]; then
    echo "Predictions directory not found: $DIR" >&2
    echo "Pass the path as the first argument, e.g.:" >&2
    echo "  bash analysis/run_calibrate.sh predictions/fold_0_mps" >&2
    exit 1
fi

python analysis/isotonic_calibration.py --val-probs "$DIR/val_probs.npy" --val-labels "$DIR/val_labels.npy" --val-mask "$DIR/val_mask.npy" --test-probs "$DIR/test_probs.npy" --test-labels "$DIR/test_labels.npy" --test-mask "$DIR/test_mask.npy" --output "$DIR/results.json"
