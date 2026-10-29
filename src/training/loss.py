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
        y_low: Tensor, 
        y_sup: Tensor, 
        target: Tensor | tuple[Tensor, Tensor]
    ) -> Tensor:

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


class IntervalQuantileLoss(nn.Module):
    """Pinball (quantile) loss for interval prediction.

    The lower output is trained as the ``alpha / 2`` quantile and the upper
    output as the ``1 - alpha / 2`` quantile.  Unlike ``IntervalRMSELoss``,
    this objective does not force both bounds toward the same point target,
    so the network has an incentive to produce a non-zero interval width.

    ``alpha`` is the expected miscoverage: ``alpha=0.1`` targets a 90% interval.
    """

    def __init__(self, alpha: float = 0.1, center_weight: float = 1.0):
        super().__init__()
        assert 0.0 < alpha < 1.0, "alpha must be in (0, 1)"
        self.alpha = alpha
        self.q_low = alpha / 2.0
        self.q_high = 1.0 - alpha / 2.0
        self.center_weight = center_weight

    @staticmethod
    def _pinball(error: Tensor, q: float) -> Tensor:
        return torch.maximum(q * error, (q - 1.0) * error)

    def forward(
        self,
        y_low: Tensor,
        y_sup: Tensor,
        target: Tensor | tuple[Tensor, Tensor],
    ) -> Tensor:
        if isinstance(target, (tuple, list)):
            t_low, t_sup = target
        else:
            t_low = t_sup = target

        loss_low = self._pinball(t_low - y_low, self.q_low).mean()
        loss_sup = self._pinball(t_sup - y_sup, self.q_high).mean()
        center = 0.5 * (y_low + y_sup)
        center_loss = torch.mean((center - t_low) ** 2)

        return loss_low + loss_sup + self.center_weight * center_loss


class WinklerLoss(nn.Module):
    """Differentiable version of the Winkler score (lower is better)."""

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        assert 0.0 < alpha < 1.0, "alpha must be in (0, 1)"
        self.alpha = alpha

    def forward(
        self,
        y_low: Tensor,
        y_sup: Tensor,
        target: Tensor | tuple[Tensor, Tensor],
    ) -> Tensor:
        if isinstance(target, (tuple, list)):
            t_low, t_sup = target
        else:
            t_low = t_sup = target

        width = (y_sup - y_low).clamp(min=0.0)
        above = torch.relu(t_low - y_sup)
        below = torch.relu(y_low - t_sup)
        return (width + (2.0 / self.alpha) * (above + below)).mean()


class SWRMSE_WS(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_low, y_sup, y_true):
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
        if y_low.dim() == 1:
            y_low = y_low.unsqueeze(1)
        if y_sup.dim() == 1:
            y_sup = y_sup.unsqueeze(1)

        ir_i = (y_true >= y_low) & (y_true <= y_sup)
        width = torch.abs(y_sup - y_low)
        center = (y_low + y_sup) / 2.0
        rmse_per_sample = torch.sqrt((y_true - center) ** 2 + self.eps)

        loss_per_sample = torch.where(ir_i, width, rmse_per_sample)

        return loss_per_sample.mean()

class SWRMSE_RMSEb(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_low, y_sup, y_true):
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
        if y_low.dim() == 1:
            y_low = y_low.unsqueeze(1)
        if y_sup.dim() == 1:
            y_sup = y_sup.unsqueeze(1)

        ir_i = (y_true >= y_low) & (y_true <= y_sup)
        
        
        center = (y_low + y_sup) / 2.0
        rmse_center = torch.sqrt((y_true - center) ** 2 + self.eps)


        rmseb = torch.sqrt((((y_true - y_low) ** 2 + (y_true - y_sup) ** 2) / 2.0) + self.eps)
        loss_per_sample = torch.where(ir_i, rmseb, rmse_center)

        return loss_per_sample.mean()

class WorstBound_RMSE(nn.Module): 
    def __init__(self,eps):
        super().__init__()
        self.eps = eps

    def forward(self,y_low,y_sup,y_true) -> float:

        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
        if y_low.dim() == 1:
            y_low = y_low.unsqueeze(1)
        if y_sup.dim() == 1:
            y_sup = y_sup.unsqueeze(1)

        rmse_low = torch.sqrt((y_true -y_low) ** 2 + self.eps)
        rmse_sup = torch.sqrt((y_true - y_sup)**2 + self.eps)

        return torch.maximum(rmse_low,rmse_sup).mean()




if __name__ == "__main__":
    
    criterion = IntervalRMSELoss(mode="combined")
    
    low = torch.tensor([[0.8], [1.9]], requires_grad=True)
    sup = torch.tensor([[1.2], [2.1]], requires_grad=True)
    target = torch.tensor([[1.0], [2.0]])

    loss = criterion((low, sup), target)
    loss.backward()
    print(f"Test IntervalRMSELoss : {loss.item():.4f}")
