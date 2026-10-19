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
        
        if reel.dim() == 2:
            reel = reel.view(-1)
            
        if pred.dim() == 2:
            lower = pred[:, 0]
            upper = pred[:, 1]
        else:
            # Handle case where pred might be [N, 1, 2] or similar
            pred = pred.squeeze()
            lower = pred[:, 0]
            upper = pred[:, 1]
        
        score = upper - lower
        score += (2/self.alpha) * (reel - upper) * (reel > upper)
        score += (2/self.alpha) * (lower - reel) * (lower > reel)
        
        return score.mean().item()



class MAE_RMSE_IR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred:list[list[float]], reel:list[float]) -> list[float]:
        pred, reel = torch.as_tensor(pred, dtype=torch.float), torch.as_tensor(reel, dtype=torch.float)
        
        # Ensure reel is 1D to match lower/upper
        if reel.dim() == 2:
            reel = reel.squeeze(-1)
            
        lower, upper = pred[:, 0], pred[:, 1]
        center = (lower + upper)/2
        
        MAE = (reel - center).abs()
        RMSE = (reel - center)**2
        inside = (reel >= lower) & (reel <= upper)
        
        return [MAE.mean().item(), RMSE.mean().item()**0.5,
                MAE[~inside].sum().item()/len(reel), (RMSE[~inside].sum().item()/len(reel))**0.5,
                inside.float().mean().item()]

