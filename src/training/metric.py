import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class WinklerScore(nn.Module):
    def __init__(self, alpha:float):
        super().__init__()

        self.alpha = alpha

    def forward(self, pred:list[list[float]], reel:list[float])-> float:
        pred, reel = torch.as_tensor(pred, dtype=torch.float), torch.as_tensor(reel, dtype=torch.float)
        pred, reel = pred.cpu(), reel.cpu()
        
        reel = reel.reshape(-1)
        if pred.dim() > 2:
            pred = pred.reshape(pred.shape[0], -1)
        if pred.dim() == 1:
            pred = pred.reshape(-1, 2)

        lower = pred[:, 0]
        upper = pred[:, 1]
        
        score = upper - lower
        score += (2/self.alpha) * (reel - upper) * (reel > upper)
        score += (2/self.alpha) * (lower - reel) * (lower > reel)
        
        return score.mean().item()


class MAE_RMSE_IR(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor | list, target: torch.Tensor | list
    ) -> tuple[float, float, float, float, float]:
        pred = torch.as_tensor(pred, dtype=torch.float32, device="cpu")
        target = torch.as_tensor(target, dtype=torch.float32, device="cpu")

        target = target.view(-1)

        if pred.dim() > 2:
            pred = pred.view(pred.shape[0], -1)
        elif pred.dim() == 1:
            pred = pred.view(-1, 2)

        lower: torch.Tensor = pred[:, 0]
        upper: torch.Tensor = pred[:, 1]
        center: torch.Tensor = (lower + upper) / 2.0

        errors: torch.Tensor = target - center
        abs_errors: torch.Tensor = errors.abs()
        sq_errors: torch.Tensor = errors.pow(2)

        is_inside: torch.Tensor = (target >= lower) & (target <= upper)
        is_outside: torch.Tensor = ~is_inside

        num_samples: int = len(target)

        mae: float = abs_errors.mean().item()
        rmse: float = sq_errors.mean().sqrt().item()

        oob_mae: float = abs_errors[is_outside].sum().item() / num_samples
        oob_rmse: float = (sq_errors[is_outside].sum().item() / num_samples) ** 0.5
        in_range_rate: float = is_inside.float().mean().item()

        return mae, rmse, oob_mae, oob_rmse, in_range_rate

class IntervalWidth(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: list[list[float]], reel: list[float] = None) -> float:
        pred = torch.as_tensor(pred, dtype=torch.float)
        pred = pred.cpu()

        if pred.dim() > 2:
            pred = pred.reshape(pred.shape[0], -1)
        if pred.dim() == 1:
            pred = pred.reshape(-1, 2)

        lower = pred[:, 0]
        upper = pred[:, 1]

        width = upper - lower
        return width.mean().item()




