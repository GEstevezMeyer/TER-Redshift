import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from src.utils.Interval import Interval
class BaseIntervallLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *outputs: Tensor | Interval, target: Tensor) -> Tensor:
        pass


class SampleWiseRMSEWidthSwitchLoss(BaseIntervallLoss):
    """
    Implémentation de Loss_A (Sample-wise RMSE / Width switch) :
    - Si target in [y_low, y_sup] (IR_i = 1) : pénalité = |y_sup - y_low|
    - Sinon (IR_i = 0)                       : pénalité = sqrt((target - c_i)^2 + eps)
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), "reduction doit être 'mean', 'sum' ou 'none'"
        self.reduction = reduction
        self.eps = eps

        def forward(self, *outputs: Tensor | Interval, target: Tensor) -> Tensor:
            # Accepte soit un objet Interval unique, soit deux tenseurs (y_low, y_sup)
            if len(outputs) == 1 and isinstance(outputs[0], Interval):
                interval = outputs[0]
            elif len(outputs) == 2:
                interval = Interval(outputs[0], outputs[1])
            else:
                raise ValueError("Fournir soit une instance d'Interval, soit deux tenseurs (y_low, y_sup).")

            inside = interval.contains(target)
            out_error = torch.sqrt((target - interval.center) ** 2 + self.eps)

            loss = torch.where(inside, interval.width, out_error)

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss


if __name__ == "__main__":
    criterion = SampleWiseRMSEWidthSwitchLoss()

    low = torch.tensor([[0.8], [1.9]], requires_grad=True)
    sup = torch.tensor([[1.2], [2.1]], requires_grad=True)
    target = torch.tensor([[1.0], [2.5]])

    # Test avec passage direct des tenseurs
    loss_tensors = criterion(low, sup, target=target)
    print(f"Loss (tenseurs bruts) : {loss_tensors.item():.4f}")

    # Test avec passage d'un objet Interval
    interval_obj = Interval(low, sup)
    loss_obj = criterion(interval_obj, target=target)
    print(f"Loss (objet Interval) : {loss_obj.item():.4f}")

    loss_obj.backward()
    print("Backward effectué avec succès.")