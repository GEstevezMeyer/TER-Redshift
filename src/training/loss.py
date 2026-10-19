import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSELoss(nn.Module):
    """
    RMSE classique : sqrt( mean( (y_pred - y_true)^2 ) + eps )
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse = F.mse_loss(y_pred, y_true, reduction=self.reduction)
        return torch.sqrt(mse + self.eps)


class IntervalRMSELoss(nn.Module):
    """
    RMSE adaptée aux sorties par intervalles (y_low, y_sup) de MacSumDenseMinMax.
    
    Modes disponibles :
    - 'combined' : sqrt( mean( (y_low - t_low)^2 + (y_sup - t_sup)^2 ) )
                   (Racine directe de la loss que tu utilises dans architecture.py)
    - 'separate' : RMSE(y_low, t_low) + RMSE(y_sup, t_sup)
    - 'midpoint' : RMSE sur le centre de l'intervalle ((y_low + y_sup) / 2)
    """
    def __init__(self, mode: str = "combined", eps: float = 1e-8):
        super().__init__()
        assert mode in ("combined", "separate", "midpoint"), \
            "mode doit être 'combined', 'separate' ou 'midpoint'"
        self.mode = mode
        self.eps = eps

    def forward(
        self, 
        preds: tuple[Tensor, Tensor], 
        target: Tensor | tuple[Tensor, Tensor]
    ) -> Tensor:
        y_low, y_sup = preds

        # Si target est un tuple (target_low, target_sup) ou un tenseur unique
        if isinstance(target, (tuple, list)):
            t_low, t_sup = target
        else:
            t_low = t_sup = target

        if self.mode == "combined":
            mse_total = torch.mean((y_low - t_low) ** 2 + (y_sup - t_sup) ** 2)
            return torch.sqrt(mse_total + self.eps)

        elif self.mode == "separate":
            rmse_low = torch.sqrt(F.mse_loss(y_low, t_low) + self.eps)
            rmse_sup = torch.sqrt(F.mse_loss(y_sup, t_sup) + self.eps)
            return rmse_low + rmse_sup

        elif self.mode == "midpoint":
            y_mid = 0.5 * (y_low + y_sup)
            t_mid = 0.5 * (t_low + t_sup)
            return torch.sqrt(F.mse_loss(y_mid, t_mid) + self.eps)


if __name__ == "__main__":
    # Petit test rapide
    criterion = IntervalRMSELoss(mode="combined")
    
    low = torch.tensor([[0.8], [1.9]], requires_grad=True)
    sup = torch.tensor([[1.2], [2.1]], requires_grad=True)
    target = torch.tensor([[1.0], [2.0]])

    loss = criterion((low, sup), target)
    loss.backward()
    print(f"Test IntervalRMSELoss : {loss.item():.4f}")
