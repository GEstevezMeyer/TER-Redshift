import csv
from pathlib import Path

from src.utils.plots import (
    METRIC_KEYS,
    VALIDATION_DATASET,
    get_dataset_names,
    get_datasets,
    get_names,
    load_results,
    plot_all_metrics,
    plot_metric,
)


def main():
    res_dir = Path("./res")
    out_root = Path("./res_plots")
    out_root.mkdir(parents=True, exist_ok=True)

    results = load_results(res_dir)
    summary_rows = []

    for run_name, data in results.items():
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

                                                                            
                                                                                
        names = get_dataset_names(data)
        split_validation = VALIDATION_DATASET in names and len(names) > 1
        exclude_validation = {VALIDATION_DATASET} if split_validation else set()

                                                                    
        for key in METRIC_KEYS:
            plot_metric(run_name, data, run_dir, key, exclude=exclude_validation)
            if split_validation:
                plot_metric(
                    run_name,
                    data,
                    run_dir,
                    key,
                    include={VALIDATION_DATASET},
                    filename_suffix=f"_{VALIDATION_DATASET}",
                )

                                                                     
        plot_all_metrics(run_name, data, run_dir, exclude=exclude_validation)
        if split_validation:
            plot_all_metrics(
                run_name,
                data,
                run_dir,
                include={VALIDATION_DATASET},
                filename_suffix=f"_{VALIDATION_DATASET}",
            )

        loss_name, metric_name = get_names(data)
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
                                                                                
        fieldnames = []
        for row in summary_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        csv_path = out_root / "results_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
