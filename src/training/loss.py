import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseIntervallLoss(nn.Module, ABC):
    def __init__(self):
        super(BaseIntervallLoss, self).__init__()

    @abstractmethod
    def forward(self, *outputs, target):
        pass

# Example de susage:
# class MyIntervallLoss(BaseIntervallLoss):
#     def __init__(self, alpha=1.0):
#         super().__init__()
#         self.alpha = alpha
#
#     def forward(self, *outputs, target):
#         lower, upper = outputs
#         loss = torch.mean(((lower + upper) / 2 - target)**2) * self.alpha
#         return loss
