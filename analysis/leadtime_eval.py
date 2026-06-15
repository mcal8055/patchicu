"""Lead-time-stratified discrimination for PatchICU per-timestep predictions.

Decomposes the single pooled AUROC that ``isotonic_calibration.py`` reports
into how discrimination behaves as a function of time relative to sepsis
onset. This is the standard early-warning diagnostic (cf. DeepAISE, NAVOY,
the NEJM AI "evaluation before onset of treatment" line of work): a near-
perfect pooled AUROC under a +/-window label is often carried by easy,
florid, at-/post-onset timesteps, while the clinically valuable pre-onset
slice is weaker. Stratifying by lead time tells the two apart.

What it computes, per dataset, aggregated across the nested-CV repetitions:
  - pooled        : AUROC/AUPRC over all real labeled timesteps. This should
                    reproduce the headline number, validating the read path.
  - pre_onset     : positives strictly before onset vs the shared negative pool
  - onset_onward  : positives at/after onset vs the shared negative pool
  - lead_<L>h     : the case's prediction at exactly onset - L hours vs the
                    shared negative pool, the "predict L hours ahead" curve.
                    L <= window are labeled positives; larger L are pre-window
                    timesteps the model still scored (flagged "extrapolated").

Onset recovery (rising-edge method): with a +/-``window``-hour label, the first
real labeled timestep that is positive is onset - window. Onset column is
therefore first_positive_col + window/resolution. Case stays whose very first
real timestep is already positive are left-truncated (onset ambiguous, should
be ~0 under the excl7 onset-exclusion) and are dropped from the stratified
analyses but still counted toward the pooled baseline. The count is reported so
you can confirm excl7 did its job.

AUROC is invariant to monotone calibration, so this reads the RAW test probs;
no calibrator or val split is needed.

Aggregation unit: the nested-CV repetition. The 5 folds within a repetition
partition the cohort, so concatenating a repetition's folds gives one full-
cohort pass with each stay scored once. Each repetition yields one estimate;
we report mean +/- sample std across repetitions. (Pooling all 25 folds would
duplicate every stay once per repetition and understate variance.)

Usage:
    python analysis/leadtime_eval.py \\
        --predictions-root predictions/all_folds_cpu_eicu \\
        --predictions-root predictions/all_folds_cpu_hirid \\
        --output predictions/leadtime_eval.json

Pass one or more --predictions-root; a basename-derived label tags each. With
two or more, a side-by-side comparison of the key strata is printed.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_repetition(fold_dir):
    """Pull the repetition index from a .../repetition_<R>/fold_<F> path."""
    m = re.search(r"repetition_(\d+)", str(fold_dir))
    if not m:
        raise ValueError(f"no repetition_<N> in path: {fold_dir}")
    return int(m.group(1))


def discover_folds(root):
    """Return {repetition_index: [fold_dir, ...]} for fold dirs holding arrays."""
    root = Path(root)
    by_rep = defaultdict(list)
    for fold_dir in sorted(root.glob("repetition_*/fold_*")):
        if (fold_dir / "test_probs.npy").exists():
            by_rep[parse_repetition(fold_dir)].append(fold_dir)
    if not by_rep:
        raise SystemExit(
            f"No fold dirs with test_probs.npy under {root}. "
            "Expected repetition_*/fold_*/test_probs.npy."
        )
    return by_rep


def process_stay(prob_row, label_row, mask_row, offset_steps, lead_steps):
    """Classify one stay and extract its scores.

    Args:
        prob_row, label_row, mask_row: 1-D arrays over the time axis for one stay.
        offset_steps: window/resolution, i.e. columns from onset-window to onset.
        lead_steps: dict {lead_hours: lead_in_columns} for the lead curve.

    Returns a dict with:
        neg            : negative-class probs (real, label 0) from this stay
        pos_all        : all positive-class probs (real, label 1) from this stay
        is_case        : bool, stay ever positive
        left_trunc     : bool, first real timestep already positive (onset unclear)
        pre, post      : pre-onset / onset-onward positive probs (cases only)
        lead           : {lead_hours: prob at onset-lead} where that column is real
    """
    valid = mask_row.astype(bool) & (label_row >= 0)
    out = {
        "neg": np.empty(0), "pos_all": np.empty(0), "is_case": False,
        "left_trunc": False, "pre": np.empty(0), "post": np.empty(0), "lead": {},
    }
    vcols = np.flatnonzero(valid)
    if vcols.size == 0:
        return out

    is_pos = label_row[vcols] > 0.5
    out["neg"] = prob_row[vcols[~is_pos]]
    if not is_pos.any():
        return out  # control stay

    out["is_case"] = True
    pos_cols = vcols[is_pos]
    out["pos_all"] = prob_row[pos_cols]
    first_pos_col = int(pos_cols[0])

    # Left-truncated: positive at the very first real timestep -> the pre-onset
    # window was clipped, so onset is ambiguous. Keep for pooled, drop from strata.
    if first_pos_col == int(vcols[0]):
        out["left_trunc"] = True
        return out

    onset_col = first_pos_col + offset_steps
    out["pre"] = prob_row[pos_cols[pos_cols < onset_col]]
    out["post"] = prob_row[pos_cols[pos_cols >= onset_col]]

    n_cols = prob_row.shape[0]
    for lead_h, steps in lead_steps.items():
        c = onset_col - steps
        if 0 <= c < n_cols and valid[c]:
            out["lead"][lead_h] = float(prob_row[c])
    return out


def auc_block(pos, neg, min_pos):
    """AUROC/AUPRC for one stratum, or None if too few positives / no negatives."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size < min_pos or neg.size == 0:
        return None
    y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    s = np.concatenate([pos, neg])
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
        "prevalence": float(pos.size / (pos.size + neg.size)),
    }


def evaluate_repetition(fold_dirs, split, offset_steps, lead_steps, min_pos):
    """Concatenate a repetition's folds (one full-cohort pass) and score strata."""
    neg, pos_all, pre, post = [], [], [], []
    lead_pos = defaultdict(list)
    n_case = n_control = n_trunc = 0

    for fold_dir in fold_dirs:
        probs = np.load(fold_dir / f"{split}_probs.npy")
        labels = np.load(fold_dir / f"{split}_labels.npy")
        mask = np.load(fold_dir / f"{split}_mask.npy")
        for i in range(probs.shape[0]):
            r = process_stay(probs[i], labels[i], mask[i], offset_steps, lead_steps)
            neg.append(r["neg"])
            if not r["is_case"]:
                n_control += 1
                continue
            n_case += 1
            pos_all.append(r["pos_all"])
            if r["left_trunc"]:
                n_trunc += 1
                continue
            pre.append(r["pre"])
            post.append(r["post"])
            for lead_h, v in r["lead"].items():
                lead_pos[lead_h].append(v)

    neg = np.concatenate(neg) if neg else np.empty(0)
    pos_all = np.concatenate(pos_all) if pos_all else np.empty(0)
    pre = np.concatenate(pre) if pre else np.empty(0)
    post = np.concatenate(post) if post else np.empty(0)

    strata = {
        "pooled": auc_block(pos_all, neg, min_pos),
        "pre_onset": auc_block(pre, neg, min_pos),
        "onset_onward": auc_block(post, neg, min_pos),
    }
    for lead_h in lead_steps:
        strata[f"lead_{lead_h}h"] = auc_block(
            np.asarray(lead_pos.get(lead_h, [])), neg, min_pos
        )
    return {
        "strata": strata,
        "n_cases": n_case, "n_controls": n_control, "n_left_truncated": n_trunc,
    }


def aggregate(per_rep, stratum_keys):
    """Mean +/- sample std across repetitions for each stratum."""
    agg = {}
    for key in stratum_keys:
        blocks = [r["strata"][key] for r in per_rep if r["strata"].get(key)]
        if not blocks:
            agg[key] = None
            continue
        aurocs = np.array([b["auroc"] for b in blocks])
        auprcs = np.array([b["auprc"] for b in blocks])
        agg[key] = {
            "auroc_mean": float(aurocs.mean()),
            "auroc_std": float(aurocs.std(ddof=1)) if aurocs.size > 1 else float("nan"),
            "auprc_mean": float(auprcs.mean()),
            "auprc_std": float(auprcs.std(ddof=1)) if auprcs.size > 1 else float("nan"),
            "n_pos_mean": float(np.mean([b["n_pos"] for b in blocks])),
            "n_neg_mean": float(np.mean([b["n_neg"] for b in blocks])),
            "prevalence_mean": float(np.mean([b["prevalence"] for b in blocks])),
            "n_reps_with_stratum": int(len(blocks)),
        }
    return agg


def evaluate_dataset(root, label, split, window_hours, resolution_hours,
                     leads, min_pos):
    offset_steps = int(round(window_hours / resolution_hours))
    lead_steps = {L: int(round(L / resolution_hours)) for L in leads}
    by_rep = discover_folds(root)

    per_rep = [
        evaluate_repetition(by_rep[r], split, offset_steps, lead_steps, min_pos)
        for r in sorted(by_rep)
    ]
    stratum_keys = ["pooled", "pre_onset", "onset_onward"] + [
        f"lead_{L}h" for L in leads
    ]
    return {
        "label": label,
        "predictions_root": str(root),
        "split": split,
        "window_hours": window_hours,
        "resolution_hours": resolution_hours,
        "leads_hours": list(leads),
        "in_window_lead_max_hours": window_hours,
        "n_repetitions": len(per_rep),
        "n_folds": sum(len(by_rep[r]) for r in by_rep),
        "n_cases_total": sum(r["n_cases"] for r in per_rep),
        "n_controls_total": sum(r["n_controls"] for r in per_rep),
        "n_left_truncated_total": sum(r["n_left_truncated"] for r in per_rep),
        "strata": aggregate(per_rep, stratum_keys),
    }


def _fmt(block):
    if block is None:
        return "    n/a (too few positives)"
    return (
        f"AUROC {block['auroc_mean']:.4f} +/- {block['auroc_std']:.4f}   "
        f"AUPRC {block['auprc_mean']:.4f}   "
        f"prev {block['prevalence_mean']:.4f}   "
        f"n_pos~{block['n_pos_mean']:.0f}/rep"
    )


def print_dataset(d):
    print(f"\n{'=' * 78}")
    print(f"DATASET: {d['label']}   (split={d['split']}, {d['n_repetitions']} reps, "
          f"{d['n_folds']} folds)")
    print(f"  cases={d['n_cases_total']}  controls={d['n_controls_total']}  "
          f"left_truncated={d['n_left_truncated_total']} "
          f"(onset-ambiguous, excluded from strata)")
    print(f"  onset = first positive timestep + {d['window_hours']}h "
          f"(rising-edge, {d['resolution_hours']}h resolution)")
    print("-" * 78)
    s = d["strata"]
    print(f"  pooled (= headline check) : {_fmt(s['pooled'])}")
    print(f"  pre_onset  (before onset) : {_fmt(s['pre_onset'])}")
    print(f"  onset_onward (at/after)   : {_fmt(s['onset_onward'])}")
    print("  lead-time curve (case prediction at onset - L hours vs negatives):")
    for L in d["leads_hours"]:
        tag = "" if L <= d["in_window_lead_max_hours"] else "  [extrapolated, pre-window]"
        print(f"    L={L:>2}h : {_fmt(s[f'lead_{L}h'])}{tag}")


def print_comparison(datasets):
    if len(datasets) < 2:
        return
    print(f"\n{'=' * 78}")
    print("COMPARISON (AUROC mean across repetitions)")
    print("-" * 78)
    keys = ["pooled", "pre_onset", "onset_onward"] + [
        f"lead_{L}h" for L in datasets[0]["leads_hours"]
    ]
    header = "  stratum".ljust(22) + "".join(d["label"][:14].rjust(16) for d in datasets)
    print(header)
    for key in keys:
        row = f"  {key}".ljust(22)
        for d in datasets:
            b = d["strata"].get(key)
            row += (f"{b['auroc_mean']:.4f}".rjust(16)) if b else "n/a".rjust(16)
        print(row)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--predictions-root", required=True, action="append", dest="roots",
        help="Dir holding repetition_*/fold_*/<split>_*.npy. Repeatable to "
             "compare datasets (e.g. all_folds_cpu_eicu, all_folds_cpu_hirid).",
    )
    ap.add_argument(
        "--label", action="append", dest="labels", default=None,
        help="Label per --predictions-root (in the same order). "
             "Defaults to the root's basename.",
    )
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="Which split's arrays to read (default: test).")
    ap.add_argument("--window-hours", type=float, default=6.0,
                    help="Label half-window around onset; onset = first "
                         "positive + this many hours (default: 6).")
    ap.add_argument("--resolution-hours", type=float, default=1.0,
                    help="Hours per timestep column (default: 1.0 for the 1hr "
                         "cohort; use e.g. 0.0333 for 2-min).")
    ap.add_argument("--leads", default="0,1,2,3,4,5,6,9,12",
                    help="Comma-separated lead times in hours for the curve "
                         "(default: 0,1,2,3,4,5,6,9,12).")
    ap.add_argument("--min-pos", type=int, default=10,
                    help="Minimum positives to report a stratum (default: 10).")
    ap.add_argument("--output", default=None,
                    help="JSON output path (default: <first root>/leadtime_eval.json).")
    args = ap.parse_args()

    leads = [int(x) if float(x).is_integer() else float(x)
             for x in args.leads.split(",") if x.strip() != ""]
    labels = args.labels or [Path(r).name for r in args.roots]
    if len(labels) != len(args.roots):
        raise SystemExit("--label count must match --predictions-root count")

    datasets = []
    for root, label in zip(args.roots, labels):
        print(f"Evaluating {label} <- {root} ...")
        datasets.append(evaluate_dataset(
            root, label, args.split, args.window_hours, args.resolution_hours,
            leads, args.min_pos,
        ))

    for d in datasets:
        print_dataset(d)
    print_comparison(datasets)

    out = args.output or str(Path(args.roots[0]) / "leadtime_eval.json")
    Path(out).write_text(json.dumps(
        {"datasets": datasets,
         "config": {"split": args.split, "window_hours": args.window_hours,
                    "resolution_hours": args.resolution_hours, "leads": leads,
                    "min_pos": args.min_pos}},
        indent=2,
    ))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
