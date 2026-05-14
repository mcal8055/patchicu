#!/usr/bin/env bash
# Convenience wrapper for deterministic inference on a trained PatchICU fold.
# Run with: bash analysis/run_inference.sh [fold_index] [device] [batch_size]
#
# Defaults: fold_index=0, device=mps, batch_size=512.
# device=cpu if MPS misbehaves; lower batch_size further (e.g. 256) if MPS OOMs.

set -e

FOLD_IDX="${1:-0}"
DEVICE="${2:-mps}"
BATCH="${3:-512}"

# Paths can be overridden via env vars (e.g. for CHPC). See run_all_folds.sh.
RUN_BASE="${RUN_BASE:-../yaib_logs/sepsis_1hr_hirid/BinaryClassification/PatchTST_1hr_30trial/2026-04-21T08-15-22}"
DATA="${DATA:-../YAIB-cohorts/data/sepsis/hirid}"
FOLD="$RUN_BASE/repetition_0/fold_${FOLD_IDX}"
OUT="predictions/fold_${FOLD_IDX}_${DEVICE}"

mkdir -p "$OUT"

# MPS fallback: routes unsupported ops to CPU automatically.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Tee output to a log file in the output dir for debugging tracebacks.
python analysis/extract_predictions.py --fold-dir "$FOLD" --data-dir "$DATA" --output-dir "$OUT" --device "$DEVICE" --batch-size "$BATCH" 2>&1 | tee "$OUT/run.log"
