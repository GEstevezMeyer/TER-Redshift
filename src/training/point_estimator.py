import torch
import torch.nn as nn


class MeanEstim(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bounds):
        low, up = bounds
        return (low + up) / 2


class MedianEstim(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bounds):
        low, up = bounds
        return (low + up) / 2

