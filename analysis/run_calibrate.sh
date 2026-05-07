#!/usr/bin/env bash
# Run isotonic calibration + scoring on a directory of saved MC predictions.
# Run with: bash analysis/run_calibrate.sh [predictions_dir]
#
# Default: mc_predictions/fold_0_n5_mps (matches default of run_inference.sh)

set -e

DIR="${1:-mc_predictions/fold_0_n5_mps}"

if [ ! -d "$DIR" ]; then
    echo "Predictions directory not found: $DIR" >&2
    echo "Pass the path as the first argument, e.g.:" >&2
    echo "  bash analysis/run_calibrate.sh mc_predictions/fold_0_n50_mps" >&2
    exit 1
fi

python analysis/mc_calibration.py --val-probs "$DIR/val_probs.npy" --val-labels "$DIR/val_labels.npy" --test-probs "$DIR/test_probs.npy" --test-labels "$DIR/test_labels.npy" --output "$DIR/results.json"
