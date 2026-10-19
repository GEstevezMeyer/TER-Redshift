import torch
import torch.nn as nn
<<<<<<< HEAD
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

=======
from abc import ABC, abstractmethod
from typing import Any
>>>>>>> 454aeec7d31b6b79317fee17a0764b647b69480d

class BaseMetric(nn.Module, ABC):
    def __init__(self):
        super(BaseMetric, self).__init__()

    @abstractmethod
<<<<<<< HEAD
    def __call__(self, model: nn.Module, dataloader: DataLoader) -> float:
        pass
        
=======
    def __call__(self, model: nn.Module, dataloader: Any) -> float:
        pass
>>>>>>> 454aeec7d31b6b79317fee17a0764b647b69480d
