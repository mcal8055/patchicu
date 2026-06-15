"""Compute timestep-level ROC and PR curves across the full 5×5 nested-CV test set.

Companion to ``compute_threshold_sweep.py``. Where that script produces
*stay-level* operating-point metrics, this one produces the *timestep-level*
discrimination curves (ROC + precision-recall) that govern the threshold
conversation — the PR curve in particular, since at a ~2.3% base rate the
ROC looks excellent while precision is the real constraint.

Runs on CHPC where per-fold ``test_probs.npy``/``test_labels.npy``/
``test_mask.npy`` arrays live. For each of the 25 (rep, fold) test splits:

  1. Applies the demo's pooled mask-aware isotonic calibrator to that fold's
     raw test probabilities (matches what the deployed demo shows — not the
     v1.1.0 per-fold calibrators used for the headline metrics). ROC/PR are
     rank metrics so calibration leaves them essentially unchanged; we
     calibrate so the marked operating points use the same calibrated
     thresholds (multiples of the base rate) as the threshold table.
  2. Restricts to real (mask=True) timesteps and computes the ROC and PR
     curves, AUROC, and average precision (AUPRC) on that fold's own split.
  3. Records where each candidate Alert threshold (a multiple of the base
     rate) lands on the curves, so the curve panel and the threshold table
     reference the same operating points.

Curves are interpolated onto a common grid per fold, then aggregated as
mean ± SE across the 25 folds — the natural unit of cross-validation
uncertainty, and consistent with how the headline scalars are reported.

IMPORTANT — folds are aggregated, not pooled. The 5 repetitions reuse the
same patients across repetitions, so concatenating all 25 folds' predictions
into one array would double-count patients and give a falsely tight curve.
We compute one curve per fold and average them with a band instead.

The output JSON contains ONLY aggregated curve arrays and scalars — no
per-stay arrays, no identifiers, no timing anchors. Safe to ship in the demo
under the HiRID DUA's no-redistribution clause.

Usage (on CHPC):

    cd /scratch/general/vast/u1561737/patchicu/patchicu
    source ../patchicu_env/bin/activate
    python analysis/compute_curve_analysis.py \\
        --predictions-dir predictions/all_folds_cpu_hirid \\
        --calibrator demo/calibrator.pkl \\
        --output demo/real_trajectories/curve_analysis.json

Then rsync ``demo/real_trajectories/curve_analysis.json`` back to your laptop.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# Must match streamlit_app.py / fit_calibrator.py / compute_threshold_sweep.py:
BASELINE_RATE = 0.0233
LABEL_HALF_WINDOW = 6

# Operating points to mark on the curves — same multipliers the threshold
# table highlights, so the two panels reference identical cutoffs.
DEFAULT_THRESHOLD_MULTIPLIERS = (1.0, 3.0, 5.0, 10.0, 20.0)


def _agg_scalar(values: list) -> tuple:
    """Mean, SD, and SE across the aggregation units (repetitions), NaN-safe."""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float("nan"), float("nan")
    sd = float(arr.std(ddof=1))
    return float(arr.mean()), sd, sd / float(np.sqrt(arr.size))


def _agg_grid(stack: np.ndarray) -> tuple:
    """Elementwise mean, SD, and SE down a [n_units, n_grid] stack."""
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    if n > 1:
        sd = stack.std(axis=0, ddof=1)
        se = sd / np.sqrt(n)
    else:
        sd = np.full(stack.shape[1], np.nan)
        se = sd
    return mean.tolist(), sd.tolist(), se.tolist()


def _compute_fold_curves(
    probs_cal: np.ndarray,    # [N_stays, T] calibrated
    labels: np.ndarray,        # [N_stays, T]
    mask: np.ndarray,          # [N_stays, T] bool
    thresholds: tuple,
    fpr_grid: np.ndarray,
    recall_grid: np.ndarray,
    baseline_rate: float,
) -> dict:
    """ROC + PR curves and operating points for ONE fold (timestep-level)."""
    valid = mask.astype(bool)
    y_true = labels[valid].astype(np.int64)
    y_score = probs_cal[valid].astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None  # degenerate fold, cannot form a curve

    # --- ROC: interpolate TPR onto the shared FPR grid -------------------
    fpr, tpr, _ = roc_curve(y_true, y_score)
    tpr_on_grid = np.interp(fpr_grid, fpr, tpr)
    auroc = float(roc_auc_score(y_true, y_score))

    # --- PR: interpolate precision onto the shared recall grid -----------
    # precision_recall_curve returns recall descending; sort ascending for
    # np.interp. The trailing (recall=0, precision=1) sentinel is dropped by
    # the sort/interp naturally.
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    order = np.argsort(recall)
    prec_on_grid = np.interp(recall_grid, recall[order], precision[order])
    auprc = float(average_precision_score(y_true, y_score))

    # --- Operating points at each calibrated threshold -------------------
    op = {}
    for mult in thresholds:
        thr = mult * baseline_rate
        pred = y_score >= thr
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        op[float(mult)] = {
            "tpr": tp / n_pos,                       # = recall
            "fpr": fp / n_neg,
            "precision": (tp / (tp + fp)) if (tp + fp) else float("nan"),
        }

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / (n_pos + n_neg),
        "auroc": auroc,
        "auprc": auprc,
        "tpr_on_grid": tpr_on_grid,
        "prec_on_grid": prec_on_grid,
        "operating_points": op,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--predictions-dir", required=True, type=Path,
                    help="Root with repetition_*/fold_*/test_*.npy")
    ap.add_argument("--calibrator", type=Path, default=None,
                    help="Optional calibrator pickle (demo/calibrator.pkl). Omit "
                         "for a cohort with no fitted calibrator (e.g. eICU) — "
                         "ROC/PR are rank metrics so curves are unaffected; only "
                         "the calibrated operating-point markers need it.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Path to write the aggregated curve JSON")
    ap.add_argument("--baseline-rate", default="0.0233",
                    help="Operating-point base rate. A float (HiRID = 0.0233), or "
                         "'auto' to use this cohort's empirical timestep prevalence "
                         "(use 'auto' for eICU).")
    ap.add_argument("--dataset-label", default="",
                    help="Human label embedded in the output (e.g. 'HiRID', 'eICU').")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=list(DEFAULT_THRESHOLD_MULTIPLIERS),
                    help="Threshold multipliers (vs base rate) to mark on the curves")
    ap.add_argument("--n-grid", type=int, default=200,
                    help="Number of points on the shared FPR / recall grids")
    args = ap.parse_args()

    if args.calibrator is not None:
        with args.calibrator.open("rb") as f:
            calibrator = pickle.load(f)
        print(f"Loaded calibrator from {args.calibrator}")
    else:
        calibrator = None
        print("No calibrator supplied — using RAW probabilities (curves "
              "unaffected; operating-point thresholds are on the raw scale).")

    fold_dirs = sorted(
        args.predictions_dir.glob("repetition_*/fold_*"),
        key=lambda p: (p.parent.name, p.name),
    )
    if not fold_dirs:
        raise SystemExit(f"No fold dirs under {args.predictions_dir}")
    print(f"Discovered {len(fold_dirs)} fold directories")

    valid_dirs = [fd for fd in fold_dirs
                  if all((fd / n).exists() for n in
                         ("test_probs.npy", "test_labels.npy", "test_mask.npy"))]

    # Resolve the base rate. 'auto' = pooled empirical timestep prevalence over
    # all folds (a cheap labels+mask pre-pass), so eICU's operating points are
    # anchored to eICU's real prevalence instead of HiRID's hardcoded 2.33%.
    if str(args.baseline_rate).lower() == "auto":
        tot_pos = tot_all = 0
        for fd in valid_dirs:
            lab = np.load(fd / "test_labels.npy")
            msk = np.load(fd / "test_mask.npy").astype(bool)
            y = lab[msk]
            tot_pos += int((y == 1).sum())
            tot_all += int(y.size)
        baseline_rate = tot_pos / tot_all if tot_all else float("nan")
        print(f"Auto base rate = {baseline_rate*100:.3f}% "
              f"({tot_pos}/{tot_all} positive timesteps)")
    else:
        baseline_rate = float(args.baseline_rate)
        print(f"Base rate = {baseline_rate*100:.3f}% (supplied)")

    thresholds = tuple(args.thresholds)
    fpr_grid = np.linspace(0.0, 1.0, args.n_grid)
    recall_grid = np.linspace(0.0, 1.0, args.n_grid)

    per_fold = []
    for fd in fold_dirs:
        need = ["test_probs.npy", "test_labels.npy", "test_mask.npy"]
        if not all((fd / n).exists() for n in need):
            print(f"  SKIP {fd.relative_to(args.predictions_dir)}: missing arrays")
            continue
        probs_raw = np.load(fd / "test_probs.npy")
        labels = np.load(fd / "test_labels.npy")
        mask = np.load(fd / "test_mask.npy").astype(bool)
        if calibrator is not None:
            probs_cal = calibrator.transform(
                probs_raw.reshape(-1)).reshape(probs_raw.shape)
        else:
            probs_cal = probs_raw
        c = _compute_fold_curves(probs_cal, labels, mask, thresholds,
                                 fpr_grid, recall_grid, baseline_rate)
        if c is None:
            print(f"  SKIP {fd.relative_to(args.predictions_dir)}: degenerate (no pos/neg)")
            continue
        c["rep"] = fd.parent.name  # e.g. "repetition_0" — the independent unit
        per_fold.append(c)
        print(f"  {fd.relative_to(args.predictions_dir)}: "
              f"AUROC={c['auroc']:.4f} AUPRC={c['auprc']:.4f} "
              f"prev={c['prevalence']*100:.2f}%")

    if not per_fold:
        raise SystemExit("No usable folds — nothing to aggregate.")

    # --- Aggregate: vertically average folds WITHIN each repetition, then
    # aggregate across repetitions. ----------------------------------------
    # Within one repetition the 5 outer folds partition the patients once, so
    # the folds are NOT independent replicates — a √(25) error bar would be
    # too tight. The 5 repetitions ARE the independent unit of uncertainty:
    # we vertically average each repetition's folds into one curve, then take
    # the spread across repetitions. Everything stays vertical averaging
    # (Provost et al. 1998; Hogan & Adams 2023) — never pooling, which would
    # assume cross-fold score comparability and bias the curve. The mean curve
    # is unchanged vs. flat fold-averaging when folds-per-rep is constant; only
    # the band becomes honest.
    from collections import defaultdict
    by_rep = defaultdict(list)
    for c in per_fold:
        by_rep[c["rep"]].append(c)
    reps = sorted(by_rep)
    print(f"\nAggregating {len(per_fold)} folds → {len(reps)} repetitions "
          f"({', '.join(f'{r}:{len(by_rep[r])}' for r in reps)})…")

    def _rep_grid(key):
        # vertical average of per-fold grids within each rep -> [n_reps, n_grid]
        return np.array([np.mean([c[key] for c in by_rep[r]], axis=0) for r in reps])

    def _rep_scalar(key):
        return [float(np.mean([c[key] for c in by_rep[r]])) for r in reps]

    tpr_mean, tpr_sd, tpr_se = _agg_grid(_rep_grid("tpr_on_grid"))
    prec_mean, prec_sd, prec_se = _agg_grid(_rep_grid("prec_on_grid"))
    auroc_mean, auroc_std, auroc_se = _agg_scalar(_rep_scalar("auroc"))
    auprc_mean, auprc_std, auprc_se = _agg_scalar(_rep_scalar("auprc"))
    prev_mean, prev_sd, prev_se = _agg_scalar(_rep_scalar("prevalence"))

    op_out = []
    for mult in thresholds:
        # per-rep mean over folds (threshold averaging at a fixed calibrated
        # threshold — valid because the pooled calibrator puts all folds on a
        # common probability scale), then mean ± SD/SE across reps.
        def _op_rep(field):
            return [float(np.nanmean([c["operating_points"][float(mult)][field]
                                      for c in by_rep[r]])) for r in reps]
        tpr_m, tpr_s_sd, tpr_s_se = _agg_scalar(_op_rep("tpr"))
        fpr_m, fpr_s_sd, fpr_s_se = _agg_scalar(_op_rep("fpr"))
        prec_m, prec_s_sd, prec_s_se = _agg_scalar(_op_rep("precision"))
        op_out.append({
            "multiplier": float(mult),
            "threshold_calibrated_pct": float(mult * baseline_rate * 100),
            "tpr_mean": tpr_m, "tpr_sd": tpr_s_sd, "tpr_se": tpr_s_se,  # tpr==recall
            "fpr_mean": fpr_m, "fpr_sd": fpr_s_sd, "fpr_se": fpr_s_se,
            "precision_mean": prec_m, "precision_sd": prec_s_sd,
            "precision_se": prec_s_se,
        })

    agg = {
        "dataset_label": args.dataset_label,
        "calibration_applied": calibrator is not None,
        "n_repetitions": len(reps),
        "n_folds_total": len(per_fold),
        "aggregation_unit": "repetition",
        "baseline_rate": baseline_rate,
        "prevalence_timestep_mean": prev_mean,
        "prevalence_timestep_se": prev_se,
        "auroc": {"mean": auroc_mean, "std": auroc_std, "se": auroc_se},
        "auprc": {"mean": auprc_mean, "std": auprc_std, "se": auprc_se},
        "roc": {
            "fpr_grid": fpr_grid.tolist(),
            "tpr_mean": tpr_mean, "tpr_sd": tpr_sd, "tpr_se": tpr_se,
        },
        "pr": {
            "recall_grid": recall_grid.tolist(),
            "precision_mean": prec_mean, "precision_sd": prec_sd,
            "precision_se": prec_se,
        },
        "operating_points": op_out,
        "source": (
            f"v1.1.0 nested-CV test predictions"
            f"{' for ' + args.dataset_label if args.dataset_label else ''}; "
            + ("pooled mask-aware isotonic calibration applied. "
               if calibrator is not None else
               "RAW probabilities (no calibrator supplied). ")
            + "Timestep-level ROC + PR by VERTICAL averaging "
            "(Provost et al. 1998; Hogan & Adams 2023): per-fold curves "
            "interpolated onto a common FPR/recall grid, vertically averaged "
            "within each repetition, then aggregated across repetitions as "
            "mean ± SD (band) / SE. NOT pooled — pooling assumes cross-fold "
            "score comparability and biases the curve. Operating-point markers "
            "use fixed-threshold averaging at the Alert thresholds (multiples "
            f"of the {baseline_rate*100:.2f}% base rate)."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(agg, indent=2))
    print(f"Wrote {args.output}")
    print(f"\nAUROC {auroc_mean:.4f} ± {auroc_std:.4f}   "
          f"AUPRC {auprc_mean:.4f} ± {auprc_std:.4f}   "
          f"(SD across {len(reps)} reps; timestep prevalence {prev_mean*100:.2f}%)")
    print("\nOperating points (timestep-level, mean across reps):")
    print(f"{'mult':>6} {'thr%':>7} {'sens(TPR)':>11} {'FPR':>8} {'precision':>11}")
    for r in op_out:
        print(f"{r['multiplier']:>6.1f} {r['threshold_calibrated_pct']:>7.2f} "
              f"{r['tpr_mean']:>11.3f} {r['fpr_mean']:>8.3f} "
              f"{r['precision_mean']:>11.3f}")


if __name__ == "__main__":
    main()
