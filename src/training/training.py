import torch 
from typing import Callable
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
import json
from src.training.loss import RMSELoss, IntervalRMSELoss
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

def training(config_path:str,training_dataloader:DataLoader,eval_dataloader:DataLoader,eval_frequency:int,epochs:int,run_name:str = "results") -> ArchitectureMacSum:
    configs = import_config(config_path)
    configs_model = configs["model"]
    configs_training = configs["training"]

    model = build_MacSumIntervallArchitecture_model(configs_model)
    device:str = return_device()

    model = model.train()
    model = model.to(device)

    optimizer:Callable = get_optimizer(model.parameters(), configs_training["optimizer"])
    loss_name,loss_function = get_loss(configs_training["loss"])
    metric_name,metric_function = get_metric(configs_training["metric"])

    loss_res:list[float] = []
    metrics_res:list[float] = []
    for i in range(epochs):
        actual_loss = 0
        tqdm_dataloader = tqdm(training_dataloader, desc=f"Epoch {i}: ")
        for x, y in tqdm_dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            
            loss = loss_function(*outputs, y)
            loss.backward()


            optimizer.step()

            actual_loss += loss.detach().cpu().item()

        loss_res.append(actual_loss)

        if (i % eval_frequency == 0):
            model.eval()
            all_preds = []
            all_reels = []
            with torch.no_grad():
                for x_eval, y_eval in eval_dataloader:
                    x_eval = x_eval.to(device)
                    y_eval = y_eval.to(device)
                    outputs = model(x_eval)
                    # outputs is (low, sup)
                    pred_low, pred_sup = outputs
                    preds = torch.stack([pred_low, pred_sup], dim=1)
                    all_preds.append(preds)
                    all_reels.append(y_eval)
            
            all_preds = torch.cat(all_preds, dim=0)
            all_reels = torch.cat(all_reels, dim=0)
            
            metric = metric_function(all_preds, all_reels)
            metrics_res.append(metric)
            print(f"{metric_name}: {metric}| {loss_name}: {actual_loss}")
            model.train()

    results = {
        "loss": loss_res,
        "metrics": metrics_res,
        "config": configs
    }
    save_results(results, model, run_name)
    
    return model

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Train Choquet Intervall Architecture")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--batch_size", type=str, required=True, help="Batch size ")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--eval_freq", type=int, required=True, help="Evaluation frequency")
    parser.add_argument("--run_name", type=str, default="results", help="Name of the result files")
    
    args = parser.parse_args()
    print(f"Config: {args.config}, Epochs: {args.epochs}, Eval Frequency: {args.eval_freq}")

    train_loader, eval_loader = main_pipeline(int(args.batch_size))
    training(args.config, train_loader, eval_loader, args.eval_freq, args.epochs, args.run_name)




