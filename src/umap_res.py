"""Build UMAP figures for every run stored in ``res/``.

Launch it with::

    python -m src.umap_res

All tuning knobs are CLI options with the values below as defaults; see
``python -m src.umap_res --help``.

For each ``<run>.json`` / ``<run>.pth`` pair found in ``res/`` the script:

1. loads the latent features and targets from the training / validation
   ``*_results.npy`` files through :class:`src.training.pipeline.EspaceLatent`,
2. feeds that dataset to the model (same normalization as during training
   evaluation, including for the validation split),
3. runs the trained model to obtain ``y_low`` and ``y_sup``,
4. fits a UMAP embedding on the training samples and projects the validation
   samples into that 2-D space,
5. writes, for each split, a three-panel value figure to
   ``umap_res/<run>/train_umap.png`` / ``validation_umap.png`` and a
   separate midpoint-error figure to
   ``umap_res/<run>/train_umap_error.png`` / ``validation_umap_error.png``.

Each value figure has three panels: ``y_true``, ``y_low`` and ``y_sup``.
The error figure colors samples by ``y_true - (y_low + y_sup) / 2``, i.e.
the difference between the target and the center of the predicted interval.
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from torch.utils.data import DataLoader, Subset

from src.training.pipeline import EspaceLatent
from src.training.training import get_model

TRAIN_DATASET = "train"
VALIDATION_DATASET = "validation"

                                                         
RES_DIR = Path("res")
OUT_DIR = Path("umap_res")
TRAIN_DATA_PATH = Path("data/epoch_40_train_results.npy")
VALIDATION_DATA_PATH = Path("data/epoch_40_val_results.npy")

                        
MAX_SAMPLES = 10_000
SEED = 42
N_NEIGHBORS = 15
MIN_DIST = 0.1
BATCH_SIZE = 512

PLOT_KEYS = ["y_true", "y_low", "y_sup"]
PLOT_TITLES = {
    "y_true": "y_true (redshift)",
    "y_low": "y_low (predicted lower bound)",
    "y_sup": "y_sup (predicted upper bound)",
}

CENTER_KEY = "center"                       
ERROR_KEY = "error"                   
ERROR_CMAP = "RdBu_r"                               


@dataclass
class Split:
    """A (possibly subsampled) dataset plus the source :class:`EspaceLatent`.

    ``indices`` maps the (subsampled) latent/target arrays used for UMAP back to
    the original dataset, so model inference can use the dataset's own
    ``__getitem__`` normalization.
    """

    name: str
    latent: np.ndarray
    target: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    y_min: float
    y_max: float
    dataset: EspaceLatent
    indices: np.ndarray

    @property
    def x_range(self) -> np.ndarray:
        return _safe_range(self.x_min, self.x_max)

    @property
    def y_range(self) -> float:
        return _safe_range(self.y_min, self.y_max)


def _safe_range(low, high):
    """Return ``high - low`` with zeros replaced by one (as in training)."""
    if np.isscalar(low):
        return high - low if high - low >= 1e-6 else 1.0
    span = high - low
    return np.where(span < 1e-6, 1.0, span)


def _subsample_indices(n_samples: int, max_samples: int, rng: np.random.Generator):
    if max_samples <= 0 or n_samples <= max_samples:
        return np.arange(n_samples)
    indices = rng.choice(n_samples, max_samples, replace=False)
    indices.sort()
    return indices


def load_split(path: str | Path, name: str, max_samples: int = MAX_SAMPLES, seed: int = SEED) -> Split:
    """Load one ``*_results.npy`` file with :class:`EspaceLatent`.

    The dataset is kept on the split because it provides the exact
    normalization (and target denormalization) used at evaluation time.
    """
    dataset = EspaceLatent(path)

    latent = np.asarray(dataset.final_latent, dtype=np.float32)
    target = np.asarray(dataset.targets, dtype=np.float32)

    valid = np.flatnonzero(np.isfinite(target) & (target > 0))
    if latent.ndim == 2:
        valid = valid[np.isfinite(latent[valid]).all(axis=1)]

    if len(valid) == 0:
        raise ValueError(f"No valid samples found in {path}")

    x_min = latent[valid].min(axis=0)
    x_max = latent[valid].max(axis=0)
    y_min = float(target[valid].min())
    y_max = float(target[valid].max())

    rng = np.random.default_rng(seed)
    picked = _subsample_indices(len(valid), max_samples, rng)
    indices = valid[picked]

    return Split(
        name=name,
        latent=latent[indices],
        target=target[indices],
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        dataset=dataset,
        indices=indices,
    )


def load_model(config: dict, pth_path: str | Path, device: torch.device) -> torch.nn.Module:
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("results.json does not contain a 'model' configuration")

    model = get_model(model_config)
    state_dict = torch.load(pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_interval(model: torch.nn.Module, dataset, device: torch.device, batch_size: int = BATCH_SIZE):
    """Run the model on an :class:`EspaceLatent` (sub)dataset.

    Inputs come from ``dataset[idx]``, so they are normalized exactly as during
    training evaluation. This also applies to the validation split, whose
    statistics differ from the training split.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    lows = []
    sups = []
    with torch.no_grad():
        for x, _ in loader:
            low, sup = model(x.to(device))
            lows.append(low.detach().cpu().numpy().reshape(-1))
            sups.append(sup.detach().cpu().numpy().reshape(-1))
    if not lows:
        return np.array([]), np.array([])
    return np.concatenate(lows), np.concatenate(sups)


def denormalize(values: np.ndarray, y_min: float, y_range: float) -> np.ndarray:
    """Inverse of the ``[-1, 1]`` target normalization used in training."""
    return (values + 1.0) / 2.0 * y_range + y_min


def compute_umap(
    train: Split,
    validation: Split,
    n_neighbors: int = N_NEIGHBORS,
    min_dist: float = MIN_DIST,
    seed: int = SEED,
):
    """Fit UMAP on the training split and project validation into that space.

    Fitting only on training data avoids leaking validation information into the
    embedding used for visualization.
    """
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

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_umap_error(
    run_name: str,
    dataset_name: str,
    embedding: np.ndarray,
    error: np.ndarray,
    out_path: str | Path,
) -> Path:
    """Save a UMAP figure colored by ``y_true - (y_low + y_sup) / 2``.

    The color scale is symmetric around zero so negative errors (true value
    below the predicted center) and positive errors are directly comparable.
    """
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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_run_values(split: Split, model, device, batch_size: int = BATCH_SIZE):
    """Predict the interval for one split and return values in physical units.

    Inference goes through the split's :class:`EspaceLatent` dataset, exactly
    like the training evaluation loop. The dataset's own ``y_min`` / ``y_range``
    are used to bring the predicted bounds back to physical units.
    """
    subset = Subset(split.dataset, split.indices)
    y_low_norm, y_sup_norm = predict_interval(model, subset, device, batch_size=batch_size)

    y_min = float(split.dataset.y_min)
    y_range = _safe_range(y_min, float(split.dataset.y_max))

    y_true = split.target.astype(np.float64)
    y_low = denormalize(y_low_norm, y_min, y_range)
    y_sup = denormalize(y_sup_norm, y_min, y_range)
    center = (y_low + y_sup) / 2.0
    return {
        "y_true": y_true,
        "y_low": y_low,
        "y_sup": y_sup,
        CENTER_KEY: center,
        ERROR_KEY: y_true - center,
    }


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

    json_files = sorted(RES_DIR.glob("*.json"))
    if not json_files:
        print(f"No result files found in {RES_DIR.resolve()}")
        return

    generated = []
    for json_path in json_files:
        run_name = json_path.stem
        pth_path = RES_DIR / f"{run_name}.pth"
        if not pth_path.exists():
            warnings.warn(f"Skipping {run_name}: missing {pth_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as handle:
            results = json.load(handle)

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
