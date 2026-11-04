import json
import csv
from pathlib import Path

from src.utils.plots import (
    METRIC_KEYS,
    get_dataset_names,
    get_datasets,
    plot_metric,
    plot_all_metrics,
)


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

        # One plot per metric, with train / validation / test overlaid.
        for key in METRIC_KEYS:
            plot_metric(run_name, data, run_dir, key)
        # Bonus: all metrics in a single figure.
        plot_all_metrics(run_name, data, run_dir)

        loss_name, metric_name = _get_names(data)
        datasets = get_datasets(data)

        row = {
            "run_name": run_name,
            "loss_name": loss_name,
            "metric_name": metric_name,
        }
        for dataset_name in get_dataset_names(data):
            dataset_metrics = datasets.get(dataset_name, {})
            for key in METRIC_KEYS:
                values = dataset_metrics.get(key, [])
                row[f"{key}_last_{dataset_name}"] = values[-1] if values else None
        summary_rows.append(row)

    if summary_rows:
        # Union of keys so that runs with different datasets/legacy formats fit.
        fieldnames = []
        for row in summary_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        csv_path = out_root / "results_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
