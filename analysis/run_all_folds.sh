#!/usr/bin/env bash
# Run deterministic inference + isotonic calibration across all 25 nested-CV
# folds (5 repetitions x 5 folds). Per-fold outputs land in
#   predictions/all_folds_${DEVICE}/repetition_R/fold_F/
#
# Run with: bash analysis/run_all_folds.sh [device] [batch_size]
#
# Defaults: device=mps, batch_size=512.
# Estimated wall time: ~30 min on MPS / ~1.5h on CPU. Run in a screen/tmux
# session or background it; per-fold output is teed to that fold's run.log.

set -e

DEVICE="${1:-mps}"
BATCH="${2:-512}"

# Paths can be overridden via env vars for non-local runs (e.g. CHPC):
#   RUN_BASE=/scratch/general/vast/$USER/patchicu/yaib_logs/.../2026-04-21T08-15-22 \
#   DATA=/scratch/general/vast/$USER/patchicu/YAIB-cohorts/R/data/sepsis/hirid \
#   bash analysis/run_all_folds.sh cuda 1024
RUN_BASE="${RUN_BASE:-../yaib_logs/sepsis_1hr_hirid/BinaryClassification/PatchTST_1hr_30trial/2026-04-21T08-15-22}"
DATA="${DATA:-../YAIB-cohorts/data/sepsis/hirid}"
ROOT="predictions/all_folds_${DEVICE}"

mkdir -p "$ROOT"

export PYTORCH_ENABLE_MPS_FALLBACK=1

start_time=$(date +%s)
total=25
done=0

for REP in 0 1 2 3 4; do
    for FOLD in 0 1 2 3 4; do
        FOLD_DIR="$RUN_BASE/repetition_${REP}/fold_${FOLD}"
        OUT="$ROOT/repetition_${REP}/fold_${FOLD}"
        mkdir -p "$OUT"

        if [ -f "$OUT/results.json" ]; then
            echo "[$(date +%H:%M:%S)] SKIP rep=${REP} fold=${FOLD} (results.json already exists)"
            done=$((done+1))
            continue
        fi

        echo "[$(date +%H:%M:%S)] === rep=${REP} fold=${FOLD} ($((done+1))/$total) ==="

        # Inference. Use PIPESTATUS to capture the python exit code (tee always returns 0).
        python analysis/extract_predictions.py --fold-dir "$FOLD_DIR" --data-dir "$DATA" --output-dir "$OUT" --device "$DEVICE" --batch-size "$BATCH" 2>&1 | tee "$OUT/run.log"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -ne 0 ]; then
            echo "[$(date +%H:%M:%S)] Inference failed (exit $rc); skipping calibration. See $OUT/run.log."
            done=$((done+1))
            continue
        fi

        # Per-fold isotonic calibration (only runs if inference succeeded).
        python analysis/isotonic_calibration.py --val-probs "$OUT/val_probs.npy" --val-labels "$OUT/val_labels.npy" --val-mask "$OUT/val_mask.npy" --test-probs "$OUT/test_probs.npy" --test-labels "$OUT/test_labels.npy" --test-mask "$OUT/test_mask.npy" --output "$OUT/results.json" 2>&1 | tee -a "$OUT/run.log"

        done=$((done+1))
        elapsed=$(( $(date +%s) - start_time ))
        avg=$(( elapsed / done ))
        remaining=$(( (total - done) * avg ))
        echo "[$(date +%H:%M:%S)] Completed ${done}/${total}; elapsed ${elapsed}s; est. remaining ${remaining}s"
    done
done

echo ""
echo "All folds complete. Aggregating..."
python analysis/aggregate_results.py --root "$ROOT" --output "$ROOT/aggregated_results.json"
