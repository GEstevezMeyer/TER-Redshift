import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class BaseMetric(nn.Module, ABC):
    def __init__(self):
        super(BaseMetric, self).__init__()

    @abstractmethod
    def __call__(self, model: nn.Module, dataloader: DataLoader) -> float:
        pass
        