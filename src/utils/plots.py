import matplotlib.pyplot as plt
from pathlib import Path


DATASET_ORDER = ["train", "validation", "test"]
DATASET_STYLES = {
    "train": {"color": "tab:blue", "linestyle": "-"},
    "validation": {"color": "tab:orange", "linestyle": "--"},
    "test": {"color": "tab:green", "linestyle": ":"},
}
# Metrics stored for every dataset by the training script.
METRIC_KEYS = ["loss", "metrics", "width", "mae", "rmse", "ir", "oob_mae", "oob_rmse"]

_FALLBACK_COLORS = ["tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


def _get_names(data):
    config = data.get("config", {}) or {}
    training = config.get("training", {}) or {}
    loss_name = (training.get("loss") or {}).get("name", "loss")
    metric_name = (training.get("metric") or {}).get("name", "metric")
    return loss_name, metric_name


def _get_metric_label(data, key):
    loss_name, metric_name = _get_names(data)
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


def plot_metric(run_name, data, out_dir, key):
    """Plot one metric with the train / validation / test curves overlaid."""
    datasets = get_datasets(data)
    names = get_dataset_names(data)
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
    path = Path(out_dir) / f"{key}_{_slug(label)}_{run_name}.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_all_metrics(run_name, data, out_dir):
    """Plot every metric in a single grid, datasets overlaid in each subplot."""
    datasets = get_datasets(data)
    names = get_dataset_names(data)

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
    path = Path(out_dir) / f"all_metrics_{run_name}.png"
    plt.savefig(path)
    plt.close()
    return path


# Backwards-compatible helpers.
def plot_loss(run_name, data, out_dir):
    return plot_metric(run_name, data, out_dir, "loss")


def plot_metrics(run_name, data, out_dir):
    return plot_metric(run_name, data, out_dir, "metrics")


def plot_width(run_name, data, out_dir):
    return plot_metric(run_name, data, out_dir, "width")


def plot_combined(run_name, data, out_dir):
    return plot_all_metrics(run_name, data, out_dir)
