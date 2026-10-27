import json
import csv
from pathlib import Path

from src.utils.plots import plot_loss, plot_metrics, plot_width, plot_combined


def load_results(res_dir):
    results = {}
    res_path = Path(res_dir)
    for file in res_path.glob("*.json"):
        run_name = file.stem
        with open(file, "r") as f:
            results[run_name] = json.load(f)
    return results


def _get_names(data):
    config = data.get("config", {}) or {}
    training = config.get("training", {}) or {}
    loss_name = (training.get("loss") or {}).get("name", "loss")
    metric_name = (training.get("metric") or {}).get("name", "metric")
    return loss_name, metric_name


def main():
    res_dir = Path("./res")
    out_root = Path("./res_plots")
    out_root.mkdir(parents=True, exist_ok=True)

    results = load_results(res_dir)
    summary_rows = []

    for run_name, data in results.items():
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        plot_loss(run_name, data, run_dir)
        plot_metrics(run_name, data, run_dir)
        plot_width(run_name, data, run_dir)
        plot_combined(run_name, data, run_dir)

        loss_name, metric_name = _get_names(data)
        loss = data.get("loss", [])
        metrics = data.get("metrics", [])
        width = data.get("width", [])
        summary_rows.append({
            "run_name": run_name,
            "loss_name": loss_name,
            "metric_name": metric_name,
            "loss_last": loss[-1] if loss else None,
            "metric_last": metrics[-1] if metrics else None,
            "width_last": width[-1] if width else None,
        })

    if summary_rows:
        csv_path = out_root / "results_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()

