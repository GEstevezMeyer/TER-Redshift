import torch
from torch import Tensor
import torch.nn as nn



class MeanEstim(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bounds):
        low, up = bounds
        return (low + up) / 2


class Conjunctive(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bounds: tuple[Tensor, Tensor], parameters: Tensor) -> Tensor:
        low, up = bounds 

        low_exp = low.unsqueeze(1)
        up_exp = up.unsqueeze(1)

     
        with torch.no_grad():
            mask = (parameters >= 0).unsqueeze(0)

    
        return torch.where(mask, up_exp, low_exp)


class Disjunctive(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bounds: tuple[Tensor, Tensor], parameters: Tensor) -> Tensor:
        low, up = bounds  
        low_exp = low.unsqueeze(1)
        up_exp = up.unsqueeze(1)

        
        with torch.no_grad():
            mask = (parameters >= 0).unsqueeze(0)

        return torch.where(mask, up_exp, low_exp)