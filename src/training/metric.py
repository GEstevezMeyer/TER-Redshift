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
    def __init__(self, alpha : float, pred : list[tuple[float, float]], reel : list[float]):
        super().__init__()
        
        self.alpha = alpha
        self.pred = pred
        self.reel = reel

    def forward(self) -> float:
        score = 0
        for p, r in zip(self.pred, self.reel):
            lower, upper = p[0], p[1]
            score += upper - lower
            if r > upper:
                score += (2/self.alpha)*(r - upper)
            elif r < lower:
                score += (2/self.alpha)*(lower - r)

        return score/len(self.reel)


class MAE_RMSE_IR(nn.Module):
    def __init__(self, alpha : float, pred : list[tuple[float, float]], reel : list[float]):
        super().__init__()
        
        self.alpha = alpha
        self.pred = pred
        self.reel = reel

    def forward(self) -> list[float]:
        IR = 0
        MAE_score, RMSE_score = 0, 0
        MAE_score_excl, RMSE_score_excl = 0, 0

        for p, r in zip(self.pred, self.reel):
            lower, upper = p[0], p[1]
            center = (lower + upper)/2

            MAE = abs(r - center)
            RMSE = (r - center)**2

            MAE_score += MAE
            RMSE_score += RMSE
            
            if r < lower or r > upper:
                MAE_score_excl += MAE
                RMSE_score_excl += RMSE

            else:
                IR += 1

        return [(MAE_score)/len(self.reel), ((RMSE_score)/len(self.reel))**0.5,
                (MAE_score_excl)/len(self.reel), ((RMSE_score_excl)/len(self.reel))**0.5,
                (IR)/len(self.reel)]
