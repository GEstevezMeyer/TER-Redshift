import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class WinklerScore(nn.Module):
    def __init__(self, alpha:float):
        super().__init__()

        self.alpha = alpha

    def forward(self, pred:torch.Tensor, reel:torch.Tensor)-> float:
        # Partie reprise du fichier metric.py de la branche architecture
        if reel.dim() == 2:
            reel = reel.view(-1)

        if pred.dim() == 2:
            lower = pred[:, 0]
            upper = pred[:, 1]
        else:
            # Handle case where pred might be [N, 2, 1]
            pred = pred.view(-1, 2)
            lower = pred[:, 0]
            upper = pred[:, 1]

        score = upper - lower
        score += (2/self.alpha) * (reel - upper) * (reel > upper)
        score += (2/self.alpha) * (lower - reel) * (lower > reel)

        return score.mean().item()


class MAE_RMSE_IR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred:torch.Tensor, reel:torch.Tensor)-> list[float]:
        # Partie reprise du fichier metric.py de la branche architecture
        if reel.dim() == 2:
            reel = reel.view(-1)

        if pred.dim() == 2:
            lower = pred[:, 0]
            upper = pred[:, 1]
        else:
            # Handle case where pred might be [N, 2, 1]
            pred = pred.view(-1, 2)
            lower = pred[:, 0]
            upper = pred[:, 1]

        center = (lower + upper)/2

        MAE = (reel - center).abs()
        RMSE = (reel - center)**2
        inside = (reel >= lower) & (reel <= upper)

        return [MAE.mean().item(), RMSE.mean().item()**0.5,
            MAE[~inside].sum().item()/len(reel), (RMSE[~inside].sum().item()/len(reel))**0.5,
            inside.float().mean().item()]



if __name__ == "__main__":
    # Test
    low = torch.tensor([[0.], [0.], [0.]])
    sup = torch.tensor([[1.], [1.], [1.]])
    pred = torch.stack([low, sup], dim=1)
    reel = torch.tensor([[0.5], [2.], [3.]])

    winkler = WinklerScore(alpha=0.1)
    print(f"Test WinklerScore : {winkler(pred, reel):.4f}")

    mae, rmse, mae_excl, rmse_excl, ir = MAE_RMSE_IR()(pred, reel)
    print(f"Test MAE_RMSE_IR : MAE={mae:.4f} RMSE={rmse:.4f} MAE_excl={mae_excl:.4f} RMSE_excl={rmse_excl:.4f} IR={ir:.4f}")
