"""Build UMAP figures for every run stored in ``res/``.

Launch it with::

    python -m src.umap_res

For each ``<run>.json`` / ``<run>.pth`` pair found in ``res/`` the script:

1. loads the latent features and targets from the training / validation
   ``*_results.npy`` files,
2. normalizes the latent features exactly like :class:`EspaceLatent`,
3. runs the trained model to obtain ``y_low`` and ``y_sup``,
4. fits a single UMAP embedding on the train + validation samples, so both
   datasets share the same 2-D space,
5. writes one figure for training and one figure for validation to
   ``umap_res/<run>/train_umap.png`` and
   ``umap_res/<run>/validation_umap.png``.

Each figure has three panels: ``y_true``, ``y_low`` and ``y_sup``.
"""

from __future__ import annotations

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

from src.training.architecture import ArchitectureMacSum

TARGET_KEY = "z_true"
LATENT_KEY = "final_latent"

TRAIN_DATASET = "train"
VALIDATION_DATASET = "validation"

# Directories and data files (no command line arguments).
RES_DIR = Path("res")
OUT_DIR = Path("umap_res")
TRAIN_DATA_PATH = Path("data/epoch_40_train_results.npy")
VALIDATION_DATA_PATH = Path("data/epoch_40_val_results.npy")

# UMAP / model settings.
MAX_SAMPLES = 10_000  # per split; set to 0 to use every sample (very slow)
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


@dataclass
class Split:
    """A (possibly subsampled) dataset plus the full-split normalization."""

    name: str
    latent: np.ndarray
    target: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    y_min: float
    y_max: float

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
    """Load one ``*_results.npy`` file and subsample it for UMAP."""
    data = np.load(path, allow_pickle=True).item()

    latent = np.asarray(data[LATENT_KEY], dtype=np.float32)
    target = np.asarray(data[TARGET_KEY], dtype=np.float32)

    valid = np.isfinite(target) & (target > 0)
    if latent.ndim == 2:
        valid &= np.isfinite(latent).all(axis=1)
    latent = latent[valid]
    target = target[valid]

    if len(target) == 0:
        raise ValueError(f"No valid samples found in {path}")

    # Normalization is computed on the full split, as EspaceLatent does.
    x_min = latent.min(axis=0)
    x_max = latent.max(axis=0)
    y_min = float(target.min())
    y_max = float(target.max())

    rng = np.random.default_rng(seed)
    indices = _subsample_indices(len(target), max_samples, rng)

    return Split(
        name=name,
        latent=latent[indices],
        target=target[indices],
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def normalized_latent(split: Split, x_min=None, x_max=None) -> np.ndarray:
    """Min-max normalize latent features to [-1, 1] (training convention)."""
    x_min = split.x_min if x_min is None else np.asarray(x_min, dtype=np.float32)
    x_max = split.x_max if x_max is None else np.asarray(x_max, dtype=np.float32)
    span = _safe_range(x_min, x_max)
    return 2.0 * (split.latent - x_min) / span - 1.0


def load_model(config: dict, pth_path: str | Path, device: torch.device) -> ArchitectureMacSum:
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("results.json does not contain a 'model' configuration")

    model = ArchitectureMacSum(**model_config)
    state_dict = torch.load(pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_interval(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = BATCH_SIZE):
    """Return ``y_low`` / ``y_sup`` predictions (still normalized)."""
    lows = []
    sups = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            low, sup = model(batch)
            lows.append(low.detach().cpu().numpy().reshape(-1))
            sups.append(sup.detach().cpu().numpy().reshape(-1))
    return np.concatenate(lows), np.concatenate(sups)


def denormalize(values: np.ndarray, y_min: float, y_range: float) -> np.ndarray:
    """Inverse of the ``[-1, 1]`` target normalization used in training."""
    return (values + 1.0) / 2.0 * y_range + y_min


def compute_umap(train: Split, validation: Split):
    """Fit UMAP on train + validation so both datasets share one embedding."""
    combined = np.vstack([train.latent, validation.latent])
    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        random_state=SEED,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(combined)
    n_train = len(train.latent)
    return embedding[:n_train], embedding[n_train:]


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


def build_run_values(split: Split, model, device, x_min=None, x_max=None):
    """Predict the interval for one split and return values in physical units."""
    x_normal = normalized_latent(split, x_min=x_min, x_max=x_max)
    y_low_norm, y_sup_norm = predict_interval(model, x_normal, device)
    y_range = split.y_range
    return {
        "y_true": split.target.astype(np.float64),
        "y_low": denormalize(y_low_norm, split.y_min, y_range),
        "y_sup": denormalize(y_sup_norm, split.y_min, y_range),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading train split...")
    train_split = load_split(TRAIN_DATA_PATH, TRAIN_DATASET)
    print("Loading validation split...")
    validation_split = load_split(VALIDATION_DATA_PATH, VALIDATION_DATASET)
    print(
        f"UMAP samples: train={len(train_split.target)}, "
        f"validation={len(validation_split.target)}"
    )

    print("Fitting UMAP on train + validation...")
    train_embedding, validation_embedding = compute_umap(train_split, validation_split)

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
        normalization = results.get("normalization") or {}

        model = None
        try:
            model = load_model(config, pth_path, device)
            print(f"[{run_name}] predicting intervals...")
            train_values = build_run_values(
                train_split,
                model,
                device,
                x_min=normalization.get("x_min"),
                x_max=normalization.get("x_max"),
            )
            validation_values = build_run_values(validation_split, model, device)
        except Exception as exc:  # noqa: BLE001 - keep processing other runs
            warnings.warn(f"Skipping {run_name}: could not compute predictions ({exc})")
            continue
        finally:
            # Release the GPU memory before loading the next run's model.
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        run_dir = OUT_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        train_path = run_dir / f"{TRAIN_DATASET}_umap.png"
        validation_path = run_dir / f"{VALIDATION_DATASET}_umap.png"

        print(f"[{run_name}] writing {train_path}")
        generated.append(
            plot_umap_dataset(run_name, TRAIN_DATASET, train_embedding, train_values, train_path)
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

    print(f"Done: {len(generated)} figure(s) written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
