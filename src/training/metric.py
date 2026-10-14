import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

"""
class BaseMetric(nn.Module, ABC):
    def __init__(self):
        super(BaseMetric, self).__init__()

    @abstractmethod
    def forward(self, model: nn.Module, dataloader: DataLoader) -> float:
        pass
"""

class WinklerScore(nn.Module):
    def __init__(self, alpha:float):
        super().__init__()

        self.alpha = alpha

    def forward(self, pred:list[list[float]], reel:list[float])-> float:
        pred, reel = torch.as_tensor(pred, dtype=torch.float), torch.as_tensor(reel, dtype=torch.float)
        lower, upper = pred[:, 0], pred[:, 1]

        score = upper - lower
        score += (2/self.alpha) * (reel - upper) * (reel > upper)
        score += (2/self.alpha) * (lower - reel) * (lower > reel)

        return score.mean()


class MAE_RMSE_IR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred:list[list[float]], reel:list[float]) -> list[float]:
        pred, reel = torch.as_tensor(pred, dtype=torch.float), torch.as_tensor(reel, dtype=torch.float)
        lower, upper = pred[:, 0], pred[:, 1]
        center = (lower + upper)/2

        MAE = (reel - center).abs()
        RMSE = (reel - center)**2
        inside = (reel >= lower) & (reel <= upper)

        return [MAE.mean(), RMSE.mean()**0.5,
                MAE[~inside].sum()/len(reel), (RMSE[~inside].sum()/len(reel))**0.5,
                inside.float().mean()]
