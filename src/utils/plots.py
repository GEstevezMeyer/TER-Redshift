import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import umap

from src.training.pipeline import EspaceLatent
from src.training.training import get_model


DATASET_ORDER = ["train", "validation", "test"]
DATASET_STYLES = {
    "train": {"color": "tab:blue", "linestyle": "-"},
    "validation": {"color": "tab:orange", "linestyle": "--"},
    "test": {"color": "tab:green", "linestyle": ":"},
}
                                                          
METRIC_KEYS = ["loss", "metrics", "width", "mae", "rmse", "ir", "oob_mae", "oob_rmse"]

_FALLBACK_COLORS = ["tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]



TRAINING_DATA_DEFAULT_PATH = "data/epoch_40_train_results.npy"
VALIDATION_DATA_DEFAULT_PATH = "data/epoch_40_val_results.npy"

TARGET_KEY = "z_true"
LATENT_KEY = "final_latent"

RES_DIR = Path("res")
TRAIN_DATASET = "train"
VALIDATION_DATASET = "validation"
TRAIN_DATA_PATH = Path(TRAINING_DATA_DEFAULT_PATH)
VALIDATION_DATA_PATH = Path(VALIDATION_DATA_DEFAULT_PATH)

MAX_SAMPLES = 10_000
SEED = 42
BATCH_SIZE = 512
N_BINS = 60

PLOT_KEYS = ["y_true", "y_low", "y_sup"]
PLOT_TITLES = {
    "y_true": "y_true (redshift)",
    "y_low": "y_low (predicted lower bound)",
    "y_sup": "y_sup (predicted upper bound)",
}
HIST_COLORS = {
    "y_true": "tab:blue",
    "y_low": "tab:orange",
    "y_sup": "tab:green",
}

CENTER_KEY = "center"
ERROR_KEY = "error"


def get_names(data):
    config = data.get("config", {}) or {}
    training = config.get("training", {}) or {}
    loss_name = (training.get("loss") or {}).get("name", "loss")
    metric_name = (training.get("metric") or {}).get("name", "metric")
    return loss_name, metric_name


def _get_metric_label(data, key):
    loss_name, metric_name = get_names(data)
    return {
        "loss": loss_name,
        "metrics": metric_name,
        "width": "width",
        "mae": "mae",
        "rmse": "rmse",
        "ir": "in_range_rate",
        "oob_mae": "oob_mae",
        "oob_rmse": "oob_rmse",
    }.get(key, key)


def get_datasets(data):
    """Return ``{dataset_name: {metric_key: values}}``.

    Handles both the new multi-dataset results and the legacy format where
    only the validation metrics were stored at the top level of the JSON.
    """
    datasets = data.get("datasets")
    if isinstance(datasets, dict) and datasets:
        return datasets

    legacy = {
        key: data.get(key, [])
        for key in METRIC_KEYS
        if key != "loss"
    }
    return {"validation": legacy}


def get_dataset_names(data):
    datasets = get_datasets(data)
    names = [name for name in DATASET_ORDER if name in datasets]
    names.extend(name for name in datasets if name not in names)
    return names


def _filter_dataset_names(data, include=None, exclude=None):
    """Select datasets to plot, optionally restricting to / skipping names."""
    names = get_dataset_names(data)
    if include is not None:
        names = [name for name in names if name in include]
    if exclude is not None:
        names = [name for name in names if name not in exclude]
    return names


def _dataset_style(name, index):
    if name in DATASET_STYLES:
        return DATASET_STYLES[name]
    return {
        "color": _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)],
        "linestyle": "-",
    }


def _x_values(data, values):
    eval_epochs = data.get("eval_epochs")
    if eval_epochs and len(eval_epochs) == len(values):
        return eval_epochs
    return list(range(len(values)))


def _slug(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


def plot_metric(run_name, data, out_dir, key, include=None, exclude=None, filename_suffix=""):
    """Plot one metric with the selected datasets overlaid.

    ``include`` / ``exclude`` filter dataset names; ``filename_suffix`` is
    inserted before the run name so several variants can coexist.
    """
    datasets = get_datasets(data)
    names = _filter_dataset_names(data, include=include, exclude=exclude)
    label = _get_metric_label(data, key)

    plt.figure()
    plotted = False
    for index, name in enumerate(names):
        values = datasets.get(name, {}).get(key, [])
        if not values:
            continue
        plt.plot(
            _x_values(data, values),
            values,
            label=name,
            **_dataset_style(name, index),
        )
        plotted = True

    if not plotted:
        plt.close()
        return None

    plt.title(f"{key}_{label}_{run_name}")
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(out_dir) / f"{key}_{_slug(label)}{filename_suffix}_{run_name}.png"
    return save_figure(plt.gcf(), path)


def plot_all_metrics(run_name, data, out_dir, include=None, exclude=None, filename_suffix=""):
    """Plot every metric in a grid, selected datasets overlaid in each subplot."""
    datasets = get_datasets(data)
    names = _filter_dataset_names(data, include=include, exclude=exclude)

    keys = [key for key in METRIC_KEYS if any(datasets.get(n, {}).get(key) for n in names)]
    if not keys:
        return None

    n_cols = 3
    n_rows = (len(keys) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for ax, key in zip(axes.flat, keys):
        label = _get_metric_label(data, key)
        for index, name in enumerate(names):
            values = datasets.get(name, {}).get(key, [])
            if not values:
                continue
            ax.plot(
                _x_values(data, values),
                values,
                label=name,
                **_dataset_style(name, index),
            )
        ax.set_title(f"{key}_{label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")

    fig.suptitle(run_name)
    plt.tight_layout()
    path = Path(out_dir) / f"all_metrics{filename_suffix}_{run_name}.png"
    return save_figure(fig, path)


                               
def plot_loss(run_name, data, out_dir):
    return plot_metric(run_name, data, out_dir, "loss")


def plot_metrics(run_name, data, out_dir):
    return plot_metric(run_name, data, out_dir, "metrics")


def plot_width(run_name, data, out_dir):
    return plot_metric(run_name, data, out_dir, "width")


def plot_combined(run_name, data, out_dir):
    return plot_all_metrics(run_name, data, out_dir)


def plot_umap_prediction(x_umap:list[list[float]],x_normal:list[list[float]],y:list[float],model) -> None: 
    
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x_normal)

    plt.figure(figsize=(12,12))

    plt.plot(x_umap[:,0],x_umap[:,1])


def safe_range(low, high):
    """Return ``high - low`` with zero spans replaced by one."""
    if np.isscalar(low):
        return high - low if high - low >= 1e-6 else 1.0
    span = high - low
    return np.where(span < 1e-6, 1.0, span)


def subsample_indices(n_samples: int, max_samples: int, rng: np.random.Generator):
    if max_samples <= 0 or n_samples <= max_samples:
        return np.arange(n_samples)
    indices = rng.choice(n_samples, max_samples, replace=False)
    indices.sort()
    return indices


@dataclass
class Split:
    """A (possibly subsampled) dataset plus its source :class:`EspaceLatent`."""

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
        return safe_range(self.x_min, self.x_max)

    @property
    def y_range(self) -> float:
        return safe_range(self.y_min, self.y_max)


def load_split(
    path: str | Path,
    name: str,
    max_samples: int = MAX_SAMPLES,
    seed: int = SEED,
) -> Split:
    """Load one ``*_results.npy`` file with :class:`EspaceLatent`."""
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
    picked = subsample_indices(len(valid), max_samples, rng)
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


def predict_interval(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
):
    """Run the model on an :class:`EspaceLatent` (sub)dataset."""
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


def build_run_values(split: Split, model, device, batch_size: int = BATCH_SIZE):
    """Predict the interval for one split and return values in physical units."""
    subset = Subset(split.dataset, split.indices)
    y_low_norm, y_sup_norm = predict_interval(model, subset, device, batch_size=batch_size)

    y_min = float(split.dataset.y_min)
    y_range = safe_range(y_min, float(split.dataset.y_max))

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


def load_results(res_dir: str | Path) -> dict:
    """Load every ``<run>.json`` file in ``res_dir`` keyed by run name."""
    results = {}
    res_path = Path(res_dir)
    for file in res_path.glob("*.json"):
        with open(file, "r", encoding="utf-8") as handle:
            results[file.stem] = json.load(handle)
    return results


def save_figure(fig, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_histogram(
    run_name: str,
    dataset_name: str,
    values: dict[str, np.ndarray],
    out_path: str | Path,
    bins: int = N_BINS,
) -> Path:
    """Save one figure overlaying y_true / y_low / y_sup on shared bins."""
    samples = [np.asarray(values[key], dtype=np.float64) for key in PLOT_KEYS]
    samples = [sample[np.isfinite(sample)] for sample in samples]
    if any(sample.size == 0 for sample in samples):
        raise ValueError(f"No finite values to histogram for {run_name} / {dataset_name}")

    all_values = np.concatenate(samples)
    vmin, vmax = float(all_values.min()), float(all_values.max())
    if vmin == vmax:
        vmax = vmin + 1.0
    edges = np.linspace(vmin, vmax, bins + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    for key, sample in zip(PLOT_KEYS, samples):
        ax.hist(
            sample,
            bins=edges,
            histtype="step",
            linewidth=2,
            color=HIST_COLORS[key],
            label=PLOT_TITLES[key],
        )

    ax.set_title(f"{run_name} - {dataset_name} - prediction histograms")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return save_figure(fig, out_path)

