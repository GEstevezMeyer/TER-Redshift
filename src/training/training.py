import torch 
from typing import Callable
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
import json
import numpy as np
from src.training.loss import RMSELoss, IntervalRMSELoss
from src.training.quantile import IntervalQuantileLoss, WinklerLoss
from src.training.metric import WinklerScore, MAE_RMSE_IR

from tqdm import tqdm
from src.utils.configs import import_config, return_device
from src.training.pipeline import main_pipeline
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
    "R2": lambda x:x,
    "winkler": WinklerScore,
    "mae_rmse_ir": MAE_RMSE_IR,
}


LOSS_REGISTRY = {
    "mse": F.mse_loss,
    "rmse": RMSELoss,
    "interval_rmse": IntervalRMSELoss,
    "quantile": IntervalQuantileLoss,
    "pinball": IntervalQuantileLoss,
    "winkler_loss": WinklerLoss,
}


DEFAULT_OPTIMIZER = optim.Adam


def get_optimizer(params, optimizer_config: dict[str, any]) -> Callable:
    optimizer_config = optimizer_config.copy()
    name = optimizer_config.pop("name")
    return OPTIMIZER_REGISTRY.get(name, DEFAULT_OPTIMIZER)(params, **optimizer_config)

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


def _dataset_statistics(dataloader: DataLoader) -> dict[str, torch.Tensor]:
    """Compute input/target normalisation statistics from the training set."""
    dataset = dataloader.dataset
    x = np.asarray(dataset.final_latent)
    y = np.asarray(dataset.targets)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    return {
        "x_mean": torch.tensor(x_mean, dtype=torch.float32),
        "x_std": torch.tensor(x_std, dtype=torch.float32),
        "y_mean": torch.tensor(float(y.mean()), dtype=torch.float32),
        "y_std": torch.tensor(float(y.std()) or 1.0, dtype=torch.float32),
    }


def training(config_path:str,training_dataloader:DataLoader,eval_dataloader:DataLoader,eval_frequency:int,epochs:int,run_name:str = "results") -> ArchitectureMacSum:
    configs = import_config(config_path)
    configs_model = configs["model"]
    configs_training = configs["training"]

    torch.manual_seed(42)
    np.random.seed(42)

    model = build_MacSumIntervallArchitecture_model(configs_model)
    device:str = return_device() or "cpu"

    model = model.to(device)
    model = model.train()

    stats = _dataset_statistics(training_dataloader)
    x_mean = stats["x_mean"].to(device)
    x_std = stats["x_std"].to(device)
    y_mean = stats["y_mean"].to(device)
    y_std = stats["y_std"].to(device)

    optimizer:Callable = get_optimizer(model.parameters(), configs_training["optimizer"])
    loss_name,loss_function = get_loss(configs_training["loss"])
    metric_name,metric_function = get_metric(configs_training["metric"])

    loss_res:list[float] = []
    metrics_res:list[float] = []
    coverage_res:list[float] = []
    width_res:list[float] = []
    rmse_res:list[float] = []

    n_samples = len(training_dataloader.dataset)
    for i in range(epochs):
        actual_loss = 0.0
        tqdm_dataloader = tqdm(training_dataloader, desc=f"Epoch {i}: ")
        for x, y in tqdm_dataloader:
            x = x.to(device)
            y = y.to(device)

            x = (x - x_mean) / x_std
            y = (y - y_mean) / y_std

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
            model.eval()
            all_preds = []
            all_reels = []
            with torch.no_grad():
                for x_eval, y_eval in eval_dataloader:
                    x_eval = x_eval.to(device)
                    x_eval = (x_eval - x_mean) / x_std
                    outputs = model(x_eval)
                    pred_low, pred_sup = outputs
                    # Back to the physical redshift scale for evaluation.
                    pred_low = pred_low * y_std + y_mean
                    pred_sup = pred_sup * y_std + y_mean
                    preds = torch.stack([pred_low, pred_sup], dim=1)
                    all_preds.append(preds)
                    all_reels.append(y_eval)
            
            all_preds = torch.cat(all_preds, dim=0)
            all_reels = torch.cat(all_reels, dim=0)
            
            metric = metric_function(all_preds, all_reels)

            # Extra diagnostics: interval width, coverage and midpoint RMSE.
            low = all_preds[:, 0].reshape(-1)
            sup = all_preds[:, 1].reshape(-1)
            reel = all_reels.reshape(-1).to(low.device)
            coverage = ((reel >= low) & (reel <= sup)).float().mean().item()
            width = (sup - low).mean().item()
            rmse = torch.sqrt(torch.mean(((low + sup) / 2 - reel) ** 2)).item()

            metrics_res.append(metric)
            coverage_res.append(coverage)
            width_res.append(width)
            rmse_res.append(rmse)

            print(f"{metric_name}: {metric} | {loss_name}: {epoch_loss:.5f} | "
                  f"coverage: {coverage:.3f} | width: {width:.4f} | rmse_mid: {rmse:.4f}")
            model.train()

    results = {
        "loss": loss_res,
        "metrics": metrics_res,
        "coverage": coverage_res,
        "width": width_res,
        "rmse_mid": rmse_res,
        "config": configs,
        "normalization": {
            "x_mean": stats["x_mean"].tolist(),
            "x_std": stats["x_std"].tolist(),
            "y_mean": float(stats["y_mean"]),
            "y_std": float(stats["y_std"]),
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
    parser.add_argument("--run_name", type=str, default="results", help="Name of the result files")
    
    args = parser.parse_args()
    print(f"Config: {args.config}, Epochs: {args.epochs}, Eval Frequency: {args.eval_freq}")

    train_loader, eval_loader = main_pipeline(int(args.batch_size))
    training(args.config, train_loader, eval_loader, args.eval_freq, args.epochs, args.run_name)
