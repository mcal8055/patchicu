"""Smoke test: fast_cache=True must produce bit-identical output to fast_cache=False.

Run from repo root:
    python patchicu/tests/test_loader_fast_cache.py

Validates the partition_by-based fast cache in PredictionPolarsDataset against
the legacy per-stay filter loop. See memory/project_patchtst_dataloader_bottleneck.md.
"""
import sys
from pathlib import Path

import polars as pl
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.data.constants import DataSegment, DataSplit


def load_demo_dict(cohort_dir: Path) -> dict:
    """Build the minimal `data` dict that PredictionPolarsDataset expects."""
    dyn = pl.read_parquet(cohort_dir / "dyn.parquet")
    outc = pl.read_parquet(cohort_dir / "outc.parquet")
    return {
        DataSplit.train: {
            DataSegment.features: dyn,
            DataSegment.outcome: outc,
        }
    }


def _tensors_equal_nan_aware(a: torch.Tensor, b: torch.Tensor) -> bool:
    """torch.equal returns False if both tensors contain NaN at the same
    position (IEEE 754: NaN != NaN). Both fast and slow cache emit NaN for
    missing measurements, so use NaN-aware comparison for correctness checks.
    """
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    return torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)) and torch.equal(
        torch.isnan(a), torch.isnan(b)
    )


def compare_datasets(fast, slow) -> None:
    """Compare every item; AssertionError on first mismatch."""
    assert len(fast) == len(slow), f"len mismatch: {len(fast)} vs {len(slow)}"
    for i in range(len(fast)):
        d_f, l_f, m_f = fast[i]
        d_s, l_s, m_s = slow[i]
        assert _tensors_equal_nan_aware(d_f, d_s), (
            f"data mismatch at idx={i}: fast.shape={d_f.shape}, slow.shape={d_s.shape}"
        )
        assert _tensors_equal_nan_aware(l_f, l_s), f"labels mismatch at idx={i}"
        assert _tensors_equal_nan_aware(m_f, m_s), f"pad_mask mismatch at idx={i}"


def run_cohort(cohort_dir: Path) -> int:
    if not cohort_dir.exists():
        print(f"SKIP: cohort missing at {cohort_dir}", file=sys.stderr)
        return 0

    data = load_demo_dict(cohort_dir)
    vars_ = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label"}

    fast = PredictionPolarsDataset(data, split=DataSplit.train, vars=vars_, fast_cache=True)
    slow = PredictionPolarsDataset(data, split=DataSplit.train, vars=vars_, fast_cache=False)

    print(f"  {cohort_dir.name}: fast={len(fast)} stays, slow={len(slow)} stays")
    compare_datasets(fast, slow)
    print(f"  PASS: bit-identical across {len(fast)} stays")
    return 0


def main() -> int:
    demo_root = HERE.parent / "demo_data" / "sepsis"
    rc = 0
    for cohort in ("mimic_demo", "eicu_demo"):
        print(f"=== {cohort} ===")
        rc |= run_cohort(demo_root / cohort)
    return rc


if __name__ == "__main__":
    sys.exit(main())
