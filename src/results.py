import json
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


def main():
    res_dir = Path("./res")
    out_root = Path("./res_plots")
    out_root.mkdir(parents=True, exist_ok=True)

    results = load_results(res_dir)
    for run_name, data in results.items():
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        plot_loss(run_name, data, run_dir)
        plot_metrics(run_name, data, run_dir)
        plot_width(run_name, data, run_dir)
        plot_combined(run_name, data, run_dir)


if __name__ == "__main__":
    main()
