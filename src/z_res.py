"""Build histograms of ``y_true`` / ``y_low`` / ``y_sup`` for every run in ``res/``.

Launch it with::

    python -m src.z_res

The loading, inference and plotting helpers are reused from
:mod:`src.utils.plots`, so the predictions come from the same ``EspaceLatent``
pipeline as during training evaluation. For each run the script writes
``z_res/<run>/train_histogram.png`` and ``z_res/<run>/validation_histogram.png``.
"""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

from src.utils.plots import (
    BATCH_SIZE,
    MAX_SAMPLES,
    N_BINS,
    RES_DIR,
    SEED,
    TRAIN_DATA_PATH,
    TRAIN_DATASET,
    VALIDATION_DATA_PATH,
    VALIDATION_DATASET,
    build_run_values,
    load_model,
    load_results,
    load_split,
    plot_histogram,
)

OUT_DIR = Path("z_res")


def main():
    parser = argparse.ArgumentParser(
        description="Plot y_true / y_low / y_sup histograms for every run in res/."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_SAMPLES,
        help="maximum samples per split (0 = all samples, default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="seed used for subsampling (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="batch size for model inference (default: %(default)s)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=N_BINS,
        help="number of histogram bins (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.max_samples < 0:
        parser.error("--max-samples must be >= 0 (0 = use every sample)")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.bins <= 0:
        parser.error("--bins must be positive")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading train split...")
    train_split = load_split(
        TRAIN_DATA_PATH, TRAIN_DATASET, max_samples=args.max_samples, seed=args.seed
    )
    print("Loading validation split...")
    validation_split = load_split(
        VALIDATION_DATA_PATH, VALIDATION_DATASET, max_samples=args.max_samples, seed=args.seed
    )
    print(
        f"Histogram samples: train={len(train_split.target)}, "
        f"validation={len(validation_split.target)}"
    )

    runs = load_results(RES_DIR)
    if not runs:
        print(f"No result files found in {RES_DIR.resolve()}")
        return

    generated = []
    for run_name, results in runs.items():
        pth_path = RES_DIR / f"{run_name}.pth"
        if not pth_path.exists():
            warnings.warn(f"Skipping {run_name}: missing {pth_path}")
            continue

        config = results.get("config") or {}

        model = None
        try:
            model = load_model(config, pth_path, device)
            print(f"[{run_name}] predicting intervals...")
            train_values = build_run_values(train_split, model, device, batch_size=args.batch_size)
            validation_values = build_run_values(
                validation_split, model, device, batch_size=args.batch_size
            )
        except Exception as exc:
            warnings.warn(f"Skipping {run_name}: could not compute predictions ({exc})")
            continue
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        run_dir = OUT_DIR / run_name
        for dataset_name, values in (
            (TRAIN_DATASET, train_values),
            (VALIDATION_DATASET, validation_values),
        ):
            out_path = run_dir / f"{dataset_name}_histogram.png"
            print(f"[{run_name}] writing {out_path}")
            generated.append(
                plot_histogram(run_name, dataset_name, values, out_path, bins=args.bins)
            )

    print(f"Done: {len(generated)} figure(s) written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
