import torch
import torch.nn as nn
from torch import Tensor


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


if __name__ == "__main__":
    low = torch.tensor([[0.8], [1.9]], requires_grad=True)
    sup = torch.tensor([[1.2], [2.1]], requires_grad=True)
    target = torch.tensor([[1.0], [2.0]])

    criterion = IntervalQuantileLoss(alpha=0.1)
    loss = criterion(low, sup, target)
    loss.backward()
    print(f"Test IntervalQuantileLoss: {loss.item():.4f}")
