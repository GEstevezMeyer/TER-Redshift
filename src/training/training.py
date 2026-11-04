import torch 
from typing import Callable
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
import argparse
import sys
import json
import numpy as np
from src.training.loss import RMSELoss, IntervalRMSELoss, IntervalQuantileLoss, WinklerLoss,SWRMSE_WS,SWRMSE_RMSEb, WorstBound_RMSE
from src.training.metric import WinklerScore, GeneralMetricsEvaluation, IntervalWidth

from tqdm import tqdm
from src.utils.configs import import_config, return_device
from src.training.pipeline import main_pipeline, TEST_SIZE_DEFAULT
from src.training.architecture import ArchitectureMacSum




OPTIMIZER_REGISTRY = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adamax": optim.Adamax,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "rmsprop": optim.RMSprop,
    "nadam": optim.NAdam,
    "radam": optim.RAdam,
    "lbfgs": optim.LBFGS,
}

METRICS_REGISTRY = {
    "winkler": WinklerScore,
}


LOSS_REGISTRY = {
    "mse": F.mse_loss,
    "rmse": RMSELoss,
    "interval_rmse": IntervalRMSELoss,
    "intervalquantile": IntervalQuantileLoss,
    "pinball": IntervalQuantileLoss,
    "winkler_loss": WinklerLoss,
    "SWRMSE_WS":SWRMSE_WS,
    "SWRMSE_RMSEb": SWRMSE_RMSEb,
    "WorstBound_RMSE": WorstBound_RMSE
}

# Metrics that are computed identically on every evaluation dataset.
EVAL_METRIC_KEYS = ["loss", "metrics", "width", "mae", "rmse", "ir", "oob_mae", "oob_rmse"]


DEFAULT_OPTIMIZER = optim.Adam


_LRSchedulerBase = getattr(lr_scheduler, "LRScheduler", lr_scheduler._LRScheduler)


class StepDecayScheduler(_LRSchedulerBase):
    """Step decay (lr * gamma every `step_size` epochs) clamped to a `min_lr` floor."""

    def __init__(self, optimizer, step_size: int = 30, gamma: float = 0.1, min_lr: float = 1e-5, last_epoch: int = -1):
        self.step_size = max(1, int(step_size))
        self.gamma = float(gamma)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        decay = self.gamma ** (self.last_epoch // self.step_size)
        return [max(base * decay, min(self.min_lr, base)) for base in self.base_lrs]


SCHEDULER_REGISTRY = {
    "step": StepDecayScheduler,
    "steplr": lr_scheduler.StepLR,
    "multisteplr": lr_scheduler.MultiStepLR,
    "exponentiallr": lr_scheduler.ExponentialLR,
    "cosineannealinglr": lr_scheduler.CosineAnnealingLR,
}


def get_optimizer(params, optimizer_config: dict[str, any]) -> Callable:
    optimizer_config = optimizer_config.copy()
    name = optimizer_config.pop("name")
    return OPTIMIZER_REGISTRY.get(name, DEFAULT_OPTIMIZER)(params, **optimizer_config)


def get_scheduler(optimizer, scheduler_config: dict[str, any] | None, epochs: int) -> tuple[str, Callable | None]:
    if not scheduler_config:
        return "none", None

    scheduler_config = scheduler_config.copy()
    name = scheduler_config.pop("name")

    if name == "step" and "step_size" not in scheduler_config:
        scheduler_config["step_size"] = max(1, epochs // 3)

    # Cosine decay ends exactly at eta_min on the last epoch, so the cycle spans
    # the epochs actually run (T_max = epochs - 1 means epoch epochs-1 -> eta_min).
    if name == "cosineannealinglr" and "T_max" not in scheduler_config:
        scheduler_config["T_max"] = max(1, epochs - 1)

    scheduler_class = SCHEDULER_REGISTRY.get(name)
    if scheduler_class is None:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {sorted(SCHEDULER_REGISTRY)}")

    return name, scheduler_class(optimizer, **scheduler_config)

def get_metric(metric_config: dict[str, any]) -> tuple[str, Callable]:
    metric_config = metric_config.copy()
    name = metric_config.pop("name")
    metric_class = METRICS_REGISTRY.get(name)
    
    if isinstance(metric_class, type):
        return name, metric_class(**metric_config)
    
    return name, metric_class
def get_loss(loss_config: dict[str, any]) -> tuple[str, Callable]:
    loss_config = loss_config.copy()
    name = loss_config.pop("name")
    loss_class = LOSS_REGISTRY.get(name)
    
    if isinstance(loss_class, type):
        return name, loss_class(**loss_config)
    
    return name, loss_class

def save_results(results: dict, model: torch.nn.Module, run_name: str = "results"):
    with open(f"res/{run_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    torch.save(model.state_dict(), f"res/{run_name}.pth")

def build_MacSumIntervallArchitecture_model(configs_model: dict[str, any]) -> ArchitectureMacSum:
    return ArchitectureMacSum(**configs_model)


def _unwrap_dataset(dataset):
    """Return the underlying dataset when a Subset is provided."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def evaluate_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_function: Callable,
    metric_function: Callable,
    width_metric: Callable,
    general_metrics_metric: Callable,
    device: str,
) -> dict[str, float]:
    """Compute the same set of metrics on one dataset."""
    model.eval()
    all_preds = []
    all_reels = []
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for x_eval, y_eval in dataloader:
            x_eval = x_eval.to(device)
            y_eval = y_eval.to(device)

            outputs = model(x_eval)
            loss = loss_function(*outputs, y_eval)
            total_loss += loss.detach().cpu().item() * x_eval.size(0)
            n_samples += x_eval.size(0)

            pred_low, pred_sup = outputs
            preds = torch.stack([pred_low, pred_sup], dim=1)
            all_preds.append(preds.cpu())
            all_reels.append(y_eval.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_reels = torch.cat(all_reels, dim=0)

    metric = metric_function(all_preds, all_reels)
    width = width_metric(all_preds)
    mae, rmse, oob_mae, oob_rmse, ir = general_metrics_metric(all_preds, all_reels)

    return {
        "loss": total_loss / max(n_samples, 1),
        "metrics": metric,
        "width": width,
        "mae": mae,
        "rmse": rmse,
        "ir": ir,
        "oob_mae": oob_mae,
        "oob_rmse": oob_rmse,
    }


def training(
    config_path: str,
    training_dataloader: DataLoader,
    eval_dataloaders: dict[str, DataLoader],
    eval_frequency: int,
    epochs: int,
    run_name: str = "results",
) -> ArchitectureMacSum:
    configs = import_config(config_path)
    configs_model = configs["model"]
    configs_training = configs["training"]

    torch.manual_seed(42)
    np.random.seed(42)

    model = build_MacSumIntervallArchitecture_model(configs_model)
    device:str = return_device() or "cpu"

    model = model.to(device)
    model = model.train()

    training_dataset = _unwrap_dataset(training_dataloader.dataset)

    x_min,y_min = training_dataset.min
    x_max,y_max = training_dataset.max

    optimizer:Callable = get_optimizer(model.parameters(), configs_training["optimizer"])
    scheduler_name,scheduler = get_scheduler(optimizer, configs_training.get("scheduler"), epochs)
    loss_name,loss_function = get_loss(configs_training["loss"])
    metric_name,metric_function = get_metric(configs_training["metric"])
    width_metric = IntervalWidth()
    general_metrics_metric = GeneralMetricsEvaluation()

    loss_res:list[float] = []
    lr_res:list[float] = []

    # One history per dataset, all containing the same metric keys.
    eval_history: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key in EVAL_METRIC_KEYS}
        for name in eval_dataloaders
    }
    eval_epochs: list[int] = []
    dataset_sizes = {
        name: len(loader.dataset) for name, loader in eval_dataloaders.items()
    }

    n_samples = len(training_dataloader.dataset)
    for i in range(epochs):
        actual_loss = 0.0
        tqdm_dataloader = tqdm(training_dataloader, desc=f"Epoch {i}: ")
        for x, y in tqdm_dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = loss_function(*outputs, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            actual_loss += loss.detach().cpu().item() * x.size(0)

        epoch_loss = actual_loss / n_samples
        loss_res.append(epoch_loss)

        if (i % eval_frequency == 0) or (i == epochs - 1):
            eval_epochs.append(i)

            for dataset_name, eval_dataloader in eval_dataloaders.items():
                dataset_metrics = evaluate_dataset(
                    model,
                    eval_dataloader,
                    loss_function,
                    metric_function,
                    width_metric,
                    general_metrics_metric,
                    device,
                )

                for key in EVAL_METRIC_KEYS:
                    eval_history[dataset_name][key].append(dataset_metrics[key])

                print(
                    f"[{dataset_name}] {metric_name}: {dataset_metrics['metrics']:.4f}"
                    f" | width: {dataset_metrics['width']:.4f}"
                    f" | {loss_name}: {dataset_metrics['loss']:.5f}"
                    f" | ir: {dataset_metrics['ir']:.4f}"
                    f" | mae: {dataset_metrics['mae']:.4f}"
                    f" | rmse: {dataset_metrics['rmse']:.4f}"
                    f" | oob_mae: {dataset_metrics['oob_mae']:.4f}"
                    f" | oob_rmse: {dataset_metrics['oob_rmse']:.4f}"
                )

            print(f"lr: {optimizer.param_groups[0]['lr']:.2e}")
            model.train()

        lr_res.append(float(optimizer.param_groups[0]["lr"]))
        if scheduler is not None:
            scheduler.step()

    results = {
        # Training loss, one value per epoch.
        "loss": loss_res,
        "lr": lr_res,
        # Epoch indices at which the datasets were evaluated.
        "eval_epochs": eval_epochs,
        # Same metrics for every dataset: train / validation / test.
        "datasets": eval_history,
        "dataset_sizes": dataset_sizes,
        "config": configs,
        "scheduler": {
            "name": scheduler_name,
            "step_size": getattr(scheduler, "step_size", None),
            "gamma": getattr(scheduler, "gamma", None),
            "min_lr": getattr(scheduler, "min_lr", None),
            "T_max": getattr(scheduler, "T_max", None),
            "eta_min": getattr(scheduler, "eta_min", None),
        },
        "normalization": {
            "type": "minmax",
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
    }
    save_results(results, model, run_name)
    
    return model

     
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Train Choquet Intervall Architecture")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--batch_size", type=int, default= 32, help="Batch size ")
    parser.add_argument("--epochs", type=int, default= 10, help="Number of epochs")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluation frequency")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT, help="Fraction of the training dataset used as test set")
    parser.add_argument("--run_name", type=str, default="results", help="Name of the result files")
    
    args = parser.parse_args()
    print(f"Config: {args.config}, Epochs: {args.epochs}, Eval Frequency: {args.eval_freq}, Test Size: {args.test_size}")

    train_loader, eval_loaders = main_pipeline(int(args.batch_size), args.test_size)
    training(args.config, train_loader, eval_loaders, args.eval_freq, args.epochs, args.run_name)
