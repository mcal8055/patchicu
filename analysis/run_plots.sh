#!/usr/bin/env bash
# Generate diagnostic plots from a directory of saved predictions.
# Run with: bash analysis/run_plots.sh [predictions_dir]

set -e

DIR="${1:-predictions/fold_0_mps}"

if [ ! -d "$DIR" ]; then
    echo "Predictions directory not found: $DIR" >&2
    exit 1
fi

python analysis/make_plots.py --pred-dir "$DIR"
