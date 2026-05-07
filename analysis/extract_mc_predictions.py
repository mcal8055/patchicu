"""Extract MC-mean prediction arrays from a trained PatchICU fold checkpoint.

Standalone inference script that uses YAIB's ``preprocess_data`` to guarantee
the per-fold splits and z-scoring exactly match the original training run,
then runs Monte Carlo dropout (N forward passes with Dropout layers selectively
enabled, BatchNorm/LayerNorm in eval mode) and saves MC-mean positive-class
probabilities plus the epistemic / aleatoric uncertainty decomposition for
the val and test splits.

Usage:
    python analysis/extract_mc_predictions.py \\
        --fold-dir yaib_logs/.../PatchTST_1hr_30trial/.../repetition_0/fold_0 \\
        --data-dir YAIB-cohorts/data/sepsis/hirid \\
        --output-dir mc_predictions/fold_0 \\
        --n-samples 50

Output files (in ``--output-dir``):
    val_probs.npy       — [n_stays, time] MC-mean positive-class probabilities
    val_labels.npy      — [n_stays, time] labels in {0, 1}; -1 = masked
    val_epistemic.npy   — [n_stays, time] mutual-information / BALD score
    val_aleatoric.npy   — [n_stays, time] expected per-sample entropy
    test_probs.npy
    test_labels.npy
    test_epistemic.npy
    test_aleatoric.npy

The probs and labels arrays can be fed directly to:
    python analysis/mc_calibration.py --val-probs val_probs.npy ...

Notes:
    - Run with the YAIB Python environment active. Imports require
      ``icu_benchmarks`` to be importable (``pip install -e`` in your YAIB
      working tree, or PYTHONPATH set to its root).
    - The HiRID cohort is DUA-bound. This script reads it locally; nothing
      patient-level is written outside ``--output-dir``.
    - Compute on a MacBook (CPU): roughly 10-15 minutes per fold for N=50.
      Streaming MC aggregation keeps memory bounded by one batch of
      predictions, not by N.
"""
import argparse
import re
import sys
from pathlib import Path

import gin
import numpy as np
import torch
from torch.utils.data import DataLoader

# YAIB
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.models import PatchTST

# Local (sibling module in the same analysis/ directory). When this script is
# run directly via `python analysis/extract_mc_predictions.py`, Python adds
# analysis/ to sys.path, so a flat sibling import works.
from mc_calibration import mc_dropout_predict


def parse_fold_path(fold_dir):
    """Extract (repetition_index, fold_index) from a path like .../repetition_0/fold_2."""
    p = str(Path(fold_dir))
    rep = re.search(r'repetition_(\d+)', p)
    fold = re.search(r'fold_(\d+)', p)
    if not rep or not fold:
        raise ValueError(
            f"Couldn't parse repetition_<N>/fold_<M> from path: {fold_dir}"
        )
    return int(rep.group(1)), int(fold.group(1))


@torch.inference_mode()
def collect_predictions(model, loader, n_samples, device):
    """Iterate loader, run MC dropout per batch; return stacked arrays.

    Returns:
        probs: [total_stays, time] MC-mean positive-class probabilities
        labels: [total_stays, time] binary labels (-1 for masked timesteps)
        epistemic: [total_stays, time] BALD score (mutual information)
        aleatoric: [total_stays, time] expected per-sample entropy
    """
    means, labels, eps_unc, ale_unc = [], [], [], []
    on_mps = (str(device) == 'mps')
    for i, batch in enumerate(loader):
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]  # labels stay on CPU
        out = mc_dropout_predict(model, x, n_samples=n_samples)
        means.append(out['mean'][..., 1].cpu().numpy())  # positive-class
        eps_unc.append(out['epistemic'].cpu().numpy())
        ale_unc.append(out['aleatoric'].cpu().numpy())
        labels.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))
        # Free per-batch refs before GC; clear MPS cache between batches
        del x, out
        if on_mps:
            torch.mps.empty_cache()
        print(f"  batch {i+1} done", flush=True)
    return (
        np.concatenate(means, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(eps_unc, axis=0),
        np.concatenate(ale_unc, axis=0),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        '--fold-dir', required=True, type=Path,
        help='Path to a fold dir (must contain train_config.gin and model.ckpt)',
    )
    ap.add_argument(
        '--data-dir', required=True, type=Path,
        help='Path to the preprocessed cohort directory (dyn/sta/outc.parquet)',
    )
    ap.add_argument(
        '--output-dir', required=True, type=Path,
        help='Directory to write the .npy prediction arrays',
    )
    ap.add_argument(
        '--n-samples', type=int, default=50,
        help='Number of MC dropout passes per batch (default: 50)',
    )
    ap.add_argument(
        '--batch-size', type=int, default=1024,
        help='DataLoader batch size (default: 1024)',
    )
    ap.add_argument(
        '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
        help='Inference device (default: cpu)',
    )
    ap.add_argument(
        '--seed', type=int, default=1111,
        help='Random seed for the data split (default: 1111, matches CHPC sweep)',
    )
    args = ap.parse_args()

    config_file = args.fold_dir / "train_config.gin"
    ckpt_file = args.fold_dir / "model.ckpt"
    missing = [f for f in (config_file, ckpt_file) if not f.exists()]
    if missing:
        sys.exit(f"Missing required file(s): {missing}")
    if not args.data_dir.exists():
        sys.exit(f"Data dir not found: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rep_idx, fold_idx = parse_fold_path(args.fold_dir)
    print(f"Fold: repetition={rep_idx}, fold={fold_idx}")

    print(f"Parsing gin config: {config_file}")
    # skip_unknown=True: the train_config.gin references configurables we don't
    # need (e.g., execute_repeated_cv, tune_hyperparameters) because we're
    # bypassing the cross_validation orchestrator. We only need data and model
    # parameters bound, which are imported above.
    gin.parse_config_file(str(config_file), skip_unknown=True)

    print("Running YAIB preprocess_data (this rebuilds the fold's splits)...")
    data = preprocess_data(
        data_dir=args.data_dir,
        seed=args.seed,
        cv_repetitions=5,
        repetition_index=rep_idx,
        cv_folds=5,
        fold_index=fold_idx,
        runmode=RunMode.classification,
    )

    print("Building val and test datasets...")
    val_dataset = PredictionPolarsDataset(
        data, split='val', name='val', ram_cache=True,
    )
    test_dataset = PredictionPolarsDataset(
        data, split='test', name='test', ram_cache=True,
    )
    print(f"  val:  {len(val_dataset)} stays")
    print(f"  test: {len(test_dataset)} stays")

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    print(f"Loading checkpoint: {ckpt_file}")
    # PyTorch 2.6+ defaults torch.load(weights_only=True), which rejects the
    # non-tensor objects (e.g., icu_benchmarks.constants.RunMode enum) saved
    # in YAIB's Lightning checkpoints. Lightning passes weights_only=True
    # explicitly, so we force-override it here. We trust our own checkpoint.
    _orig_torch_load = torch.load
    def _torch_load_compat(*a, **kw):
        kw['weights_only'] = False  # force-override Lightning's explicit True
        return _orig_torch_load(*a, **kw)
    torch.load = _torch_load_compat
    try:
        model = PatchTST.load_from_checkpoint(str(ckpt_file), map_location=args.device)
    finally:
        torch.load = _orig_torch_load
    model = model.to(args.device)

    print(f"Running MC dropout (n_samples={args.n_samples}) on val...")
    val_probs, val_labels, val_eps, val_ale = collect_predictions(
        model, val_loader, args.n_samples, args.device,
    )
    print(f"Running MC dropout on test...")
    test_probs, test_labels, test_eps, test_ale = collect_predictions(
        model, test_loader, args.n_samples, args.device,
    )

    out = args.output_dir
    np.save(out / "val_probs.npy", val_probs)
    np.save(out / "val_labels.npy", val_labels)
    np.save(out / "val_epistemic.npy", val_eps)
    np.save(out / "val_aleatoric.npy", val_ale)
    np.save(out / "test_probs.npy", test_probs)
    np.save(out / "test_labels.npy", test_labels)
    np.save(out / "test_epistemic.npy", test_eps)
    np.save(out / "test_aleatoric.npy", test_ale)

    print()
    print(f"Saved to {out}/:")
    print(f"  val_probs.npy       shape={val_probs.shape}  dtype={val_probs.dtype}")
    print(f"  val_labels.npy      shape={val_labels.shape}  dtype={val_labels.dtype}")
    print(f"  val_epistemic.npy   shape={val_eps.shape}  range=[{val_eps.min():.4f}, {val_eps.max():.4f}]")
    print(f"  val_aleatoric.npy   shape={val_ale.shape}  range=[{val_ale.min():.4f}, {val_ale.max():.4f}]")
    print(f"  test_probs.npy      shape={test_probs.shape}  dtype={test_probs.dtype}")
    print(f"  test_labels.npy     shape={test_labels.shape}  dtype={test_labels.dtype}")
    print(f"  test_epistemic.npy  shape={test_eps.shape}  range=[{test_eps.min():.4f}, {test_eps.max():.4f}]")
    print(f"  test_aleatoric.npy  shape={test_ale.shape}  range=[{test_ale.min():.4f}, {test_ale.max():.4f}]")
    print()
    print("Next:")
    print(f"  python analysis/mc_calibration.py \\")
    print(f"      --val-probs {out}/val_probs.npy \\")
    print(f"      --val-labels {out}/val_labels.npy \\")
    print(f"      --test-probs {out}/test_probs.npy \\")
    print(f"      --test-labels {out}/test_labels.npy \\")
    print(f"      --output {out}/results.json")


if __name__ == '__main__':
    main()
