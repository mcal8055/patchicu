"""Aggregate per-fold isotonic calibration results into a summary JSON.

Walks ``--root`` looking for ``repetition_*/fold_*/results.json`` files
produced by ``isotonic_calibration.py``, computes mean and std across folds
for AUROC, AUPRC, Brier, and ECE (pre and post calibration), and writes a
single ``aggregated_results.json``.

Usage:
    python analysis/aggregate_results.py \\
        --root predictions/all_folds_mps \\
        --output predictions/all_folds_mps/aggregated_results.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


METRICS = ('auroc', 'auprc', 'brier', 'ece')
BLOCKS = ('pre_calibration', 'post_calibration')


def parse_rep_fold(path: Path):
    """Extract (rep, fold) ints from a path with repetition_R/fold_F components."""
    s = str(path)
    rep = re.search(r'repetition_(\d+)', s)
    fold = re.search(r'fold_(\d+)', s)
    if not rep or not fold:
        return None
    return int(rep.group(1)), int(fold.group(1))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--root', required=True, type=Path,
                    help='Directory containing repetition_*/fold_*/results.json files')
    ap.add_argument('--output', required=True, type=Path,
                    help='Path to write the aggregated JSON')
    args = ap.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f'Root does not exist: {args.root}')

    fold_files = sorted(args.root.glob('repetition_*/fold_*/results.json'))
    if not fold_files:
        raise SystemExit(
            f'No results.json files found under {args.root}/repetition_*/fold_*'
        )

    per_fold = []
    for path in fold_files:
        rep_fold = parse_rep_fold(path.parent)
        if rep_fold is None:
            continue
        rep, fold = rep_fold
        with path.open() as f:
            data = json.load(f)
        per_fold.append({
            'rep': rep,
            'fold': fold,
            'n_test_timesteps': data.get('n_test_timesteps'),
            'pre_calibration': data['pre_calibration'],
            'post_calibration': data['post_calibration'],
        })

    n = len(per_fold)
    print(f'Loaded {n} per-fold results.json file(s)')

    summary = {
        'n_folds': n,
        'fold_paths': [str(p.relative_to(args.root)) for p in fold_files],
    }
    for block in BLOCKS:
        summary[block] = {}
        for metric in METRICS:
            vals = np.array([f[block][metric] for f in per_fold])
            summary[block][metric] = {
                'mean': float(vals.mean()),
                'std': float(vals.std(ddof=1)) if n > 1 else 0.0,
                'min': float(vals.min()),
                'max': float(vals.max()),
                'values': vals.tolist(),
            }

    summary['per_fold'] = per_fold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print(f'\nWrote {args.output}')
    print()
    for block in BLOCKS:
        print(f'{block}:')
        for metric in METRICS:
            m = summary[block][metric]
            print(
                f"  {metric:<6} mean={m['mean']:.4f} "
                f"std={m['std']:.4f} "
                f"min={m['min']:.4f} max={m['max']:.4f}"
            )
        print()


if __name__ == '__main__':
    main()
