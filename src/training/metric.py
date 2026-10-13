import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any

class BaseMetric(nn.Module, ABC):
    """
    Abstract base class for evaluation metrics.
    """
    def __init__(self):
        super(BaseMetric, self).__init__()

    @abstractmethod
    def __call__(self, model: nn.Module, dataloader: Any) -> float:
        """
        Calculate the metric for the given model and dataloader.
        
        Args:
            model: The model to evaluate.
            dataloader: The evaluation dataloader.
            
        Returns:
            float: The calculated metric value.
        """
        pass
