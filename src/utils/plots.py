import matplotlib.pyplot as plt
from pathlib import Path


def _get_names(data):
    config = data.get("config", {}) or {}
    training = config.get("training", {}) or {}
    loss_name = (training.get("loss") or {}).get("name", "loss")
    metric_name = (training.get("metric") or {}).get("name", "metric")
    return loss_name, metric_name


def plot_loss(run_name, data, out_dir):
    loss = data.get("loss", [])
    if not loss:
        return
    loss_name, _ = _get_names(data)
    plt.figure()
    plt.plot(range(len(loss)), loss)
    plt.title(f"loss_{loss_name}_{run_name}")
    plt.xlabel("Epoch")
    plt.ylabel(loss_name)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"loss_{loss_name}_{run_name}.png")
    plt.close()


def plot_metrics(run_name, data, out_dir):
    metrics = data.get("metrics", [])
    if not metrics:
        return
    _, metric_name = _get_names(data)
    plt.figure()
    plt.plot(range(len(metrics)), metrics)
    plt.title(f"metrics_{metric_name}_{run_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"metrics_{metric_name}_{run_name}.png")
    plt.close()


def plot_width(run_name, data, out_dir):
    width = data.get("width", [])
    if not width:
        return
    plt.figure()
    plt.plot(range(len(width)), width)
    plt.title(f"width_{run_name}")
    plt.xlabel("Epoch")
    plt.ylabel("width")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"width_{run_name}.png")
    plt.close()


def plot_combined(run_name, data, out_dir):
    loss = data.get("loss", [])
    metrics = data.get("metrics", [])
    width = data.get("width", [])
    if not any([loss, metrics, width]):
        return
    loss_name, metric_name = _get_names(data)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if loss:
        axes[0].plot(range(len(loss)), loss)
        axes[0].set_title(f"loss_{loss_name}_{run_name}")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel(loss_name)
    if metrics:
        axes[1].plot(range(len(metrics)), metrics)
        axes[1].set_title(f"metrics_{metric_name}_{run_name}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(metric_name)
    if width:
        axes[2].plot(range(len(width)), width)
        axes[2].set_title(f"width_{run_name}")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("width")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"combined_{loss_name}_{metric_name}_{run_name}.png")
    plt.close()

