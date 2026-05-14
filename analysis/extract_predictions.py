"""Extract per-fold prediction arrays from a trained PatchICU checkpoint.

Standalone inference script that uses YAIB's ``preprocess_data`` to guarantee
the per-fold splits and z-scoring exactly match the original training run,
then runs a single deterministic forward pass and saves positive-class
probabilities, labels, and the pad_mask for the val and test splits.

The pad_mask is REQUIRED downstream: YAIB's PredictionPolarsDataset coerces
labels at padded and unlabeled timesteps to 0 (matching pad_value), which
makes any naive {0, 1} filter silently include those positions as
false-zero negatives. Saving the mask alongside probs and labels lets the
calibration script filter to real, labeled timesteps the same way YAIB's
train/eval loop does (see icu_benchmarks/models/wrappers.py:335-336).

Usage:
    python analysis/extract_predictions.py \\
        --fold-dir yaib_logs/.../PatchTST_1hr_30trial/.../repetition_0/fold_0 \\
        --data-dir YAIB-cohorts/data/sepsis/hirid \\
        --output-dir predictions/fold_0

Output files (in ``--output-dir``):
    val_probs.npy    — [n_stays, time] positive-class probabilities
    val_labels.npy   — [n_stays, time] binary labels (0 at padded positions)
    val_mask.npy     — [n_stays, time] bool, True for real labeled timesteps
    test_probs.npy
    test_labels.npy
    test_mask.npy

Feed all six to ``isotonic_calibration.py``.

Notes:
    - Run with the YAIB Python environment active. Imports require
      ``icu_benchmarks`` to be importable (``pip install -e`` in your YAIB
      working tree, or PYTHONPATH set to its root).
    - The HiRID cohort is DUA-bound. This script reads it locally; nothing
      patient-level is written outside ``--output-dir``.
    - Inference is a single deterministic forward pass (no MC dropout): the
      GP-tuned PatchICU model converged on dropout~=0, so MC sampling
      produced near-zero epistemic variance and was retired (see
      project_v02_demo_uncertainty.md). For uncertainty quantification, use
      a deep ensemble across the 25 nested-CV checkpoints instead.
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
def collect_predictions(model, loader, device):
    """Iterate loader, run a single deterministic forward pass per batch.

    Returns:
        probs:  [total_stays, time] positive-class softmax probabilities
        labels: [total_stays, time] binary labels (0 at padded positions per
                YAIB's pad_value=0.0 convention)
        masks:  [total_stays, time] bool, True for real labeled timesteps
    """
    probs, labels, masks = [], [], []
    on_mps = (str(device) == 'mps')
    model.eval()
    for i, batch in enumerate(loader):
        # YAIB's PredictionPolarsDataset returns (data, labels, pad_mask).
        # Older callers (this script's MC predecessor) dropped batch[2] and
        # then silently scored padded positions as label=0 negatives. Don't
        # repeat that mistake — pull all three.
        if len(batch) < 3:
            raise RuntimeError(
                f"Expected (data, labels, mask) 3-tuple from loader; got "
                f"{len(batch)}-tuple. Refusing to score without the mask."
            )
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]
        m = batch[2]
        logits = model(x)
        p = torch.softmax(logits, dim=-1)[..., 1].cpu().numpy()
        probs.append(p)
        labels.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))
        masks.append(m.numpy() if torch.is_tensor(m) else np.asarray(m))
        del x, logits
        if on_mps:
            torch.mps.empty_cache()
        print(f"  batch {i+1} done", flush=True)
    return (
        np.concatenate(probs, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(masks, axis=0).astype(bool),
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
    # skip_unknown=True covers unknown configurables, but not unknown
    # *parameters* of known configurables (e.g., Adam.decoupled_weight_decay
    # exists in newer PyTorch but not older). Iteratively strip such bindings
    # whenever gin complains, since they're version-drift artifacts that
    # don't affect inference.
    import re as _re
    with open(config_file) as _f:
        _config_lines = _f.readlines()
    while True:
        try:
            gin.parse_config(''.join(_config_lines), skip_unknown=True)
            break
        except ValueError as _e:
            _m = _re.search(
                r"Configurable '(\w+)' doesn't have a parameter named '(\w+)'",
                str(_e),
            )
            if not _m:
                raise
            _cfg, _param = _m.group(1), _m.group(2)
            _pat = _re.compile(rf'^\s*{_re.escape(_cfg)}\.{_re.escape(_param)}\s*=')
            _before = len(_config_lines)
            _config_lines = [_l for _l in _config_lines if not _pat.match(_l)]
            if len(_config_lines) == _before:
                raise
            print(
                f"  [skip] unknown gin binding {_cfg}.{_param} "
                "(version drift; safe to ignore for inference)"
            )

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
        kw['weights_only'] = False
        return _orig_torch_load(*a, **kw)
    torch.load = _torch_load_compat
    try:
        model = PatchTST.load_from_checkpoint(str(ckpt_file), map_location=args.device)
    finally:
        torch.load = _orig_torch_load
    model = model.to(args.device)

    print("Running deterministic inference on val...")
    val_probs, val_labels, val_mask = collect_predictions(model, val_loader, args.device)
    print("Running deterministic inference on test...")
    test_probs, test_labels, test_mask = collect_predictions(model, test_loader, args.device)

    out = args.output_dir
    np.save(out / "val_probs.npy", val_probs)
    np.save(out / "val_labels.npy", val_labels)
    np.save(out / "val_mask.npy", val_mask)
    np.save(out / "test_probs.npy", test_probs)
    np.save(out / "test_labels.npy", test_labels)
    np.save(out / "test_mask.npy", test_mask)

    print()
    print(f"Saved to {out}/:")
    print(f"  val_probs.npy   shape={val_probs.shape}  dtype={val_probs.dtype}")
    print(f"  val_labels.npy  shape={val_labels.shape}  dtype={val_labels.dtype}")
    print(f"  val_mask.npy    shape={val_mask.shape}  real_frac={val_mask.mean():.4f}")
    print(f"  test_probs.npy  shape={test_probs.shape}  dtype={test_probs.dtype}")
    print(f"  test_labels.npy shape={test_labels.shape}  dtype={test_labels.dtype}")
    print(f"  test_mask.npy   shape={test_mask.shape}  real_frac={test_mask.mean():.4f}")
    print()
    print("Next:")
    print(f"  python analysis/isotonic_calibration.py \\")
    print(f"      --val-probs {out}/val_probs.npy \\")
    print(f"      --val-labels {out}/val_labels.npy \\")
    print(f"      --val-mask {out}/val_mask.npy \\")
    print(f"      --test-probs {out}/test_probs.npy \\")
    print(f"      --test-labels {out}/test_labels.npy \\")
    print(f"      --test-mask {out}/test_mask.npy \\")
    print(f"      --output {out}/results.json")


if __name__ == '__main__':
    main()
