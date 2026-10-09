# Reference:
# Code Ocean, Capsule 8551785, v1.
# https://codeocean.com/capsule/8551785/tree/v1


import torch 
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F
from typing import Callable

from src.utils.configs import return_device

def sources_and_subsets_nodes(N: int) -> tuple[list[list[int]], list[list[int]]]:
    str1: str = "{0:{fill}" + str(N) + "b}"
    a: list[str] = []
    for i in range(1, 2**N):
        a.append(str1.format(i, fill='0'))

    sourcesInNode: list[list[int]] = []
    sourcesNotInNode: list[list[int]] = []
    subset: list[list[int]] = []
    sourceList: list[int] = list(range(N))

    def node_subset(node: int, sourcesInNodes: list[int]) -> list[int]:
        return [node - 2**i for i in sourcesInNodes]

    def string_to_integer_array(s: str, ch: str) -> list[int]:
        N = len(s)
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]

    for j in range(len(a)):
        idxLR = string_to_integer_array(a[j], '1')
        sourcesInNode.append(idxLR)
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))
        subset.append(node_subset(j, idxLR))

    return sourcesInNode, subset


def subset_to_indices(indices: Tensor) -> list[Tensor]:
    return [i for i in indices]


class Choquet_integral(nn.Module):
    
    N_in: int
    N_out: int
    nVars: int
    mu: nn.Parameter
    sourcesInNode: list[Tensor]
    subset: list[Tensor]
    FM: Tensor

    def __init__(self, N_in: int, N_out: int) -> None:
        super().__init__()

        self.N_in = N_in
        self.N_out = N_out
        self.nVars = 2**self.N_in - 2

        dummy: Tensor = (1 / self.N_in) * torch.ones((self.nVars, self.N_out), requires_grad=True)
        self.mu = nn.Parameter(dummy)

        sources_lists, subset_lists = sources_and_subsets_nodes(self.N_in)
        self.sourcesInNode = [torch.tensor(x) for x in sources_lists]
        self.subset = [torch.tensor(x) for x in subset_lists]

    def forward(self, inputs: Tensor) -> Tensor:
        self.FM = self.chi_nn_mu(self.mu)
        sortInputs: Tensor
        sortInd: Tensor
        sortInputs, sortInd = torch.sort(inputs, 1, True)
        M, N = inputs.size()
        sortInputs = torch.cat((sortInputs, torch.zeros(M, 1)), 1)
        sortInputs = sortInputs[:, :-1] - sortInputs[:, 1:]

        out: Tensor = torch.cumsum(torch.pow(2, sortInd), 1) - torch.ones(1, dtype=torch.int64)

        data: Tensor = torch.zeros((M, self.nVars + 1))

        for i in range(M):
            data[i, out[i, :]] = sortInputs[i, :]

        ChI: Tensor = torch.matmul(data, self.FM)

        return ChI

    def chi_nn_mu(self, chi_mu: Tensor) -> Tensor:
        chi_mu = torch.abs(chi_mu)

        FM: Tensor = chi_mu[None, 0, :]
        for i in range(1, self.nVars):
            indices: list[Tensor] = subset_to_indices(self.subset[i])
            if len(indices) == 1:
                FM = torch.cat((FM, chi_mu[None, i, :]), 0)
            else:
                maxVal, _ = torch.max(FM[indices, :], 0)
                temp: Tensor = torch.add(maxVal, chi_mu[i, :])
                FM = torch.cat((FM, temp[None, :]), 0)

        FM = torch.cat([FM, torch.ones((1, self.N_out))], 0)
        FM = torch.min(FM, torch.ones(1))

        return FM

class ChoquetInterval(torch.nn.Module):
    def __init__(self, N_in, N_out):
        super().__init__()

        self.a = Choquet_integral(N_in, N_out)
        self.b = Choquet_integral(N_in, N_out)

    def forward(self, x:Tensor) -> tuple[Tensor,Tensor]:
        lower = self.a(x)
        upper = lower + nn.functional.softplus(self.b(x))

        return upper, lower

    



class ChoquetIntervallArchitecture(nn.Module):
    def __init__(self, n_in: int, layer_dense: list[int], activation_function: Callable | list[Callable]):
        super().__init__()
        self.n_in = n_in
        self.layers_dense = layer_dense
        self.n = len(layer_dense)

        self.ChoquetInterval = ChoquetInterval(self.n_in, self.layers_dense[0])
        self.Layers_dense = nn.ModuleList([nn.Linear(self.layers_dense[i], self.layers_dense[i+1]) for i in range(self.n-1)])
        self.activation_function = [activation_function for _ in range(self.n-1)] if callable(activation_function) else activation_function

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_upper, x_lower = self.ChoquetInterval(x)

        for layer, act in zip(self.Layers_dense, self.activation_function):
            x_upper = act(layer(x_upper))
            x_lower = act(layer(x_lower))

        minimum = torch.minimum(x_upper, x_lower)
        maximum = torch.maximum(x_lower, x_upper)

        return minimum, maximum
        
        





if __name__ == "__main__": 
    
    import torch.nn.functional as F
    model = ChoquetIntervallArchitecture(n_in=3, layer_dense=[4, 8, 1], activation_function=F.tanh)
    x = torch.rand(5, 3)          

    minimum, maximum = model(x)

    print(f"x : {x}")
    print(f"minimum : {minimum}")
    print(f"maximum : {maximum}")
    print("minimum.shape:", minimum.shape)   
    print("maximum.shape:", maximum.shape)   
    print("min <= max:", bool(torch.all(minimum <= maximum)))   

    (minimum.sum() + maximum.sum()).backward()
    print("gradientes OK:", all(p.grad is not None for p in model.parameters()))