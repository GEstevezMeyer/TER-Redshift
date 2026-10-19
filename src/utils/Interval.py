import torch
from torch import Tensor

class  Interval:
    def __init__(self, y_low: Tensor, y_sup: Tensor):
        """
        Stocke les bornes de l'intervalle prédit.
        """
        self.y_low = y_low
        self.y_sup = y_sup

    @property
    def center(self) -> Tensor:
        """Calcule le centre de l'intervalle."""
        return 0.5 * (self.y_low + self.y_sup)

    @property
    def width(self) -> Tensor:
        """Calcule la largeur de l'intervalle pour chaque échantillon."""
        return torch.abs(self.y_sup - self.y_low)

    def mean_width(self) -> float:
        """Calcule la largeur moyenne sur le batch."""
        return self.width.mean().item()

    def rmse(self, target: Tensor, eps: float = 1e-8) -> float:
        """Calcule la RMSE entre la cible et le centre de l'intervalle."""
        return torch.sqrt(torch.mean((target - self.center) ** 2) + eps).item()

    def mae(self, target: Tensor) -> float:
        """Calcule la MAE entre la cible et le centre de l'intervalle."""
        return torch.mean(torch.abs(target - self.center)).item()

    def inclusion_rate(self, target: Tensor) -> float:
        """Calcule la proportion de cibles qui tombent dans l'intervalle."""
        inside = (target >= self.y_low) & (target <= self.y_sup)
        return inside.float().mean().item()