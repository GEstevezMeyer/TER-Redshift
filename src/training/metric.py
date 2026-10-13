import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any

class BaseMetric(nn.Module, ABC):
    def __init__(self):
        super(BaseMetric, self).__init__()

    @abstractmethod
    def __call__(self, model: nn.Module, dataloader: Any) -> float:
        pass
