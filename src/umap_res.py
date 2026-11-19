"""Build UMAP figures for every run stored in ``res/``.

Launch it with::

    python -m src.umap_res

All tuning knobs are CLI options with the values below as defaults; see
``python -m src.umap_res --help``.

Shared loading and inference helpers live in :mod:`src.utils.plots` and are
reused by the other plotting scripts. For each ``<run>.json`` / ``<run>.pth``
pair this script:

1. loads the latent features and targets through :class:`EspaceLatent`,
2. feeds that dataset to the model (same normalization as during training
   evaluation, including for the validation split),
3. runs the trained model to obtain ``y_low`` and ``y_sup``,
4. fits a UMAP embedding on the training samples and projects the validation
   samples into that 2-D space,
5. writes value and midpoint-error figures to ``umap_res/<run>/``.
"""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

from src.utils.plots import (
    BATCH_SIZE,
    ERROR_KEY,
    MAX_SAMPLES,
    PLOT_KEYS,
    PLOT_TITLES,
    RES_DIR,
    SEED,
    TRAIN_DATA_PATH,
    TRAIN_DATASET,
    VALIDATION_DATA_PATH,
    VALIDATION_DATASET,
    Split,
    build_run_values,
    load_model,
    load_results,
    load_split,
    save_figure,
)

OUT_DIR = Path("umap_res")

N_NEIGHBORS = 15
MIN_DIST = 0.1
ERROR_CMAP = "RdBu_r"


def compute_umap(
    train: Split,
    validation: Split,
    n_neighbors: int = N_NEIGHBORS,
    min_dist: float = MIN_DIST,
    seed: int = SEED,
):
    """Fit UMAP on the training split and project validation into that space."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        n_jobs=1,
    )
    train_embedding = reducer.fit_transform(train.latent)
    validation_embedding = reducer.transform(validation.latent)
    return train_embedding, validation_embedding


def plot_umap_dataset(
    run_name: str,
    dataset_name: str,
    embedding: np.ndarray,
    values: dict[str, np.ndarray],
    out_path: str | Path,
) -> Path:
    """Save one figure with y_true / y_low / y_sup for a single dataset."""
    all_values = np.concatenate([values[key] for key in PLOT_KEYS])
    vmin, vmax = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    if vmin == vmax:
        vmax = vmin + 1.0

    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    last_scatter = None
    for ax, key in zip(axes, PLOT_KEYS):
        last_scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=values[key],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=8,
            alpha=0.75,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(PLOT_TITLES[key])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{run_name} - {dataset_name} - UMAP of the latent space")
    fig.tight_layout()
    cbar = fig.colorbar(last_scatter, ax=axes.ravel().tolist(), shrink=0.85, pad=0.015)
    cbar.set_label("value")
    return save_figure(fig, out_path)


def plot_umap_error(
    run_name: str,
    dataset_name: str,
    embedding: np.ndarray,
    error: np.ndarray,
    out_path: str | Path,
) -> Path:
    """Save a UMAP figure colored by ``y_true - (y_low + y_sup) / 2``."""
    error = np.asarray(error, dtype=np.float64)
    finite = np.isfinite(error)
    abs_max = float(np.max(np.abs(error[finite]))) if np.any(finite) else 0.0
    if abs_max <= 0.0:
        abs_max = 1.0

    fig, ax = plt.subplots(figsize=(8, 6.5))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=error,
        cmap=plt.get_cmap(ERROR_CMAP),
        vmin=-abs_max,
        vmax=abs_max,
        s=8,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(f"{run_name} - {dataset_name} - midpoint prediction error")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label("y_true - (y_low + y_sup) / 2")

    fig.tight_layout()
    return save_figure(fig, out_path)


def main():
    parser = argparse.ArgumentParser(description="Build UMAP figures for every run stored in res/.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_SAMPLES,
        help="maximum samples per split used for UMAP (0 = all samples, default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="seed for subsampling and UMAP (default: %(default)s)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=N_NEIGHBORS,
        help="UMAP number of neighbors (default: %(default)s)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=MIN_DIST,
        help="UMAP min_dist (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="batch size for model inference (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.max_samples < 0:
        parser.error("--max-samples must be >= 0 (0 = use every sample)")
    if args.n_neighbors <= 0 or args.batch_size <= 0:
        parser.error("--n-neighbors and --batch-size must be positive")
    if args.min_dist < 0:
        parser.error("--min-dist must be >= 0")

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
        f"UMAP samples: train={len(train_split.target)}, "
        f"validation={len(validation_split.target)}"
    )

    print("Fitting UMAP on train and projecting validation...")
    train_embedding, validation_embedding = compute_umap(
        train_split,
        validation_split,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        seed=args.seed,
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
        run_dir.mkdir(parents=True, exist_ok=True)

        train_path = run_dir / f"{TRAIN_DATASET}_umap.png"
        validation_path = run_dir / f"{VALIDATION_DATASET}_umap.png"
        train_error_path = run_dir / f"{TRAIN_DATASET}_umap_error.png"
        validation_error_path = run_dir / f"{VALIDATION_DATASET}_umap_error.png"

        print(f"[{run_name}] writing {train_path}")
        generated.append(
            plot_umap_dataset(run_name, TRAIN_DATASET, train_embedding, train_values, train_path)
        )
        print(f"[{run_name}] writing {train_error_path}")
        generated.append(
            plot_umap_error(
                run_name,
                TRAIN_DATASET,
                train_embedding,
                train_values[ERROR_KEY],
                train_error_path,
            )
        )
        print(f"[{run_name}] writing {validation_path}")
        generated.append(
            plot_umap_dataset(
                run_name,
                VALIDATION_DATASET,
                validation_embedding,
                validation_values,
                validation_path,
            )
        )
        print(f"[{run_name}] writing {validation_error_path}")
        generated.append(
            plot_umap_error(
                run_name,
                VALIDATION_DATASET,
                validation_embedding,
                validation_values[ERROR_KEY],
                validation_error_path,
            )
        )

    print(f"Done: {len(generated)} figure(s) written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
