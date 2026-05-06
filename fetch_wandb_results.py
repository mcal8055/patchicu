"""Pull all runs + artifacts from the patchicu wandb project into a local dir.

Usage:
    pip install wandb  # in your yaib conda env
    wandb login        # one-time, browser-based
    python fetch_wandb_results.py

Writes:
    wandb_results/
    ├── index.json                          # summary of all runs
    ├── <run_name>/
    │   ├── config.json                     # run config (best HPs, framework)
    │   ├── summary.json                    # final metrics (flat)
    │   ├── history.csv                     # per-fold history
    │   └── artifacts/                      # raw files (aggregated_test_metrics.json, optuna DB, plots)
    │       └── ...
"""

import json
from pathlib import Path

import wandb


ENTITY = "verity-hogans-university-of-utah"
PROJECT = "patchicu"
OUT = Path(__file__).parent / "wandb_results"


def main():
    OUT.mkdir(exist_ok=True)
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    print(f"Found {len(runs)} runs in {ENTITY}/{PROJECT}")

    index = []
    for run in runs:
        safe_name = run.name.replace("/", "_") if run.name else run.id
        run_dir = OUT / safe_name
        run_dir.mkdir(exist_ok=True)
        print(f"\n=== {safe_name} ({run.state}) ===")

        # config (includes best_hps if the upload script attached them)
        with (run_dir / "config.json").open("w") as f:
            json.dump(dict(run.config), f, indent=2, default=str)

        # summary (final metrics — what we flattened into final/avg.AUC etc)
        with (run_dir / "summary.json").open("w") as f:
            json.dump(dict(run.summary), f, indent=2, default=str)

        # history — per-fold metrics logged via wandb.log()
        try:
            hist = run.history()
            hist.to_csv(run_dir / "history.csv", index=False)
            print(f"  history: {len(hist)} rows, {len(hist.columns)} cols")
        except Exception as e:
            print(f"  history failed: {e}")

        # artifacts — raw JSON/DB/PNG files we attached
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        try:
            for art in run.logged_artifacts():
                print(f"  downloading artifact: {art.name}")
                art.download(root=str(artifacts_dir / art.name.replace(":", "_")))
        except Exception as e:
            print(f"  artifact download failed: {e}")

        index.append({
            "name": safe_name,
            "id": run.id,
            "state": run.state,
            "created": str(run.created_at),
            "tags": list(run.tags),
            "config": dict(run.config),
            "summary_keys": [k for k in run.summary.keys() if not k.startswith("_")],
        })

    with (OUT / "index.json").open("w") as f:
        json.dump(index, f, indent=2, default=str)

    print(f"\n✓ Wrote {len(runs)} runs to {OUT}/")
    print(f"✓ Index at {OUT}/index.json")


if __name__ == "__main__":
    main()
