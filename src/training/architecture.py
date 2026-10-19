import torch 
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F
<<<<<<< HEAD
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
=======




class NeuroneMacSum(nn.Module):
    def __init__(self, n_in:int, n_out:int,conjunctive = True):
>>>>>>> 454aeec7d31b6b79317fee17a0764b647b69480d
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.parametres = nn.Parameter(torch.rand(self.n_out, self.n_in)) 
        self.is_conjunctive = conjunctive  


    def _mixed(self, x_upper: Tensor, x_lower: Tensor, upper_when_pos: bool):
            B, K = x_upper.shape
            with torch.no_grad():
                mask_neg = (self.parametres < 0).unsqueeze(0)  
    
            x_lower_exp = x_lower.unsqueeze(1).expand(B, K, self.n_in)
            x_upper_exp = x_upper.unsqueeze(1).expand(B, K, self.n_in)
    
            if upper_when_pos:  
                x_sup = torch.where(mask_neg, x_lower_exp, x_upper_exp)  
                x_inf = torch.where(mask_neg, x_upper_exp, x_lower_exp)  
            else:               
                x_sup = torch.where(mask_neg, x_upper_exp, x_lower_exp)  
                x_inf = torch.where(mask_neg, x_lower_exp, x_upper_exp)  
            return x_inf, x_sup
    
    def conjunctive(self, x_upper: Tensor, x_lower: Tensor):
        return self._mixed(x_upper, x_lower, upper_when_pos=False)
    
    def disjunctive(self, x_upper: Tensor, x_lower: Tensor):
        return self._mixed(x_upper, x_lower, upper_when_pos=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)                                 

        w = self.parametres                                      

<<<<<<< HEAD
            minimum = torch.minimum(x_upper, x_lower)
            maximum = torch.maximum(x_lower, x_upper)

            return minimum, maximum
=======
      
        w_dec, _ = torch.sort(w, dim=1, descending=True)
        w_cro, _ = torch.sort(w, dim=1, descending=False)
        w_plus = torch.clamp(w_dec, min=0)
        w_moin = torch.clamp(w_cro, max=0)

       
        xd, _ = torch.sort(x, dim=1, descending=True)
        xc, _ =  torch.sort(x, dim=1, descending=False)

        xd_p, xd_m = torch.clamp(xd, min=0), torch.clamp(xd, max=0)
        xc_p, xc_m = torch.clamp(xc, min=0), torch.clamp(xc, max=0)

        beta_k_sup  = torch.cummax(xd_p, dim=-1).values  + torch.cummax(xd_m, dim=-1).values
        beta_k_low  = torch.cummax(-xd_p, dim=-1).values + torch.cummax(-xd_m, dim=-1).values
        gamma_k_sup = torch.cummin(xc_p, dim=-1).values  + torch.cummin(xc_m, dim=-1).values
        gamma_k_low = torch.cummin(-xc_p, dim=-1).values + torch.cummin(-xc_m, dim=-1).values

        print(beta_k_low.shape)
      
        zeros = torch.zeros_like(beta_k_sup[..., :1])
        beta_k_1_sup  = torch.cat([zeros, beta_k_sup[..., :-1]],  dim=-1)
        beta_k_1_low  = torch.cat([zeros, beta_k_low[..., :-1]],  dim=-1)
        gamma_k_1_sup = torch.cat([zeros, gamma_k_sup[..., :-1]], dim=-1)
        gamma_k_1_low = torch.cat([zeros, gamma_k_low[..., :-1]], dim=-1)

        
        y_sup = (beta_k_sup - beta_k_1_sup) @ w_plus.T   + (gamma_k_sup - gamma_k_1_sup) @ w_moin.T
        y_low = -((beta_k_low - beta_k_1_low) @ w_plus.T + (gamma_k_low - gamma_k_1_low) @ w_moin.T)

        if squeeze:
            y_low, y_sup = y_low.squeeze(0), y_sup.squeeze(0)

        if self.conjunctive:
            y_low, y_sup = self.conjunctive(y_sup,y_low)
        else:
            y_low,y_sup = self.disjunctive(y_sup,y_low)

        return y_low, y_sup

    


class MinMax(nn.Module): 
    def __init__(self):
        super().__init__()

    def forward(self,a,b): 
        return torch.minimum(a, b), torch.maximum(a, b)

class MacSumDenseMinMax(nn.Module):
    def __init__(self, MacSumIn: int, MacSumOut: int, output_size: int, linearStart: int,
                 linearN: int, increment: int):
        super().__init__()
        self.MacSumIn = MacSumIn
        self.MacSumOut = MacSumOut

        self.linearStart = linearStart
        self.linearN = linearN
        self.increment = increment
        self.output_size = output_size

        last_size = MacSumOut - increment * linearN * (linearN + 1) // 2  

        assert linearN >= 0, "linearN can't be negative"
        assert increment >= 0, "increment can't be negative"
        assert linearStart <= MacSumOut, "linearStart can't be bigger than MacSumOut"
        assert output_size >= 1, "output_size must be positive"
        assert output_size <= last_size, "output size can't be bigger than the last linear layer input size"

        self.MacSum = NeuroneMacSum(self.MacSumIn, self.MacSumOut)
        self.MinMax = MinMax()

        listLinear = []
        current_input = self.MacSumOut

        for i in range(0, linearN):
            listLinear.append(nn.Linear(current_input, current_input - (i + 1) * increment))
            listLinear.append(nn.ReLU())                      
            current_input = current_input - (i + 1) * increment

        listLinear.append(nn.Linear(current_input, output_size))   

        self.ForwardPass = nn.Sequential(*listLinear)


    def forward(self, x):
        x_low, x_sup = self.MacSum(x)
        return self.MinMax(self.ForwardPass(x_low), self.ForwardPass(x_sup)) 

       




            



>>>>>>> 454aeec7d31b6b79317fee17a0764b647b69480d





if __name__ == "__main__": 
<<<<<<< HEAD
    import torch.nn.functional as F

    model = ChoquetIntervallArchitecture(n_in=3, layer_dense=[4, 8, 1], activation_function=F.tanh)
    x = torch.rand(5, 3)          
=======
    torch.manual_seed(0)

    print("--- TEST NEURONE ---")
    n_in, n_out = 5, 3
    neurone = NeuroneMacSum(n_in, n_out)
    x = torch.tensor([0.5, -1.0, 2.0, -0.3, 1.5])
    y_low, y_sup = neurone(x)
    print(f"y_low: {y_low}")
    print(f"y_sup: {y_sup}")
    
    opt = torch.optim.Adam(neurone.parameters(), lr=0.05)
    for epoch in range(50):
        opt.zero_grad()
        yl, ys = neurone(x)
        loss = ((ys - 1.0) ** 2 + (yl + 1.0) ** 2).sum()
        loss.backward()
        opt.step()
    print(f"Final neurone loss: {loss.item():.4f}")
>>>>>>> 454aeec7d31b6b79317fee17a0764b647b69480d

    print("\n--- TEST ARCHITECTURE ---")
    arch = MacSumDenseMinMax(MacSumIn=5, MacSumOut=8, output_size=1,
                            linearStart=8, linearN=2, increment=1, inproper=False)
    x_arch = torch.randn(5)

    optimizer_arch = torch.optim.Adam(arch.parameters(), lr=0.01)
    target = torch.tensor([1.0])

<<<<<<< HEAD
    (minimum.sum() + maximum.sum()).backward()
    print("gradientes OK:", all(p.grad is not None for p in model.parameters()))

=======
    for epoch in range(50):
        optimizer_arch.zero_grad()
        low, sup = arch(x_arch)
        loss_arch = torch.mean((low - target)**2 + (sup - target)**2)
        loss_arch.backward()
        optimizer_arch.step()

    final_low, final_sup = arch(x_arch)
    final_loss = torch.mean((final_low - target)**2 + (final_sup - target)**2)

    print(f"Final Architecture Low: {final_low.item():.4f}")
    print(f"Final Architecture Sup: {final_sup.item():.4f}")
    print(f"Final Architecture Loss: {final_loss.item():.4f}")

    assert final_low.shape == (1,) and final_sup.shape == (1,)
    assert (final_low <= final_sup).all(), "MinMax should give low <= sup"
    assert final_loss.item() < 1.0, "loss did not decrease"
    print("Architecture Test: SUCCESS")


    print("\n--- TEST ARCHITECTURE (BATCH) ---")
    torch.manual_seed(0)
    B = 32
    arch = MacSumDenseMinMax(MacSumIn=5, MacSumOut=8, output_size=1,
                            linearStart=8, linearN=2, increment=1, inproper=False)
    x_arch = torch.randn(B, 5)
    target = torch.ones(B, 1)

    optimizer_arch = torch.optim.Adam(arch.parameters(), lr=0.01)

    with torch.no_grad():
        low0, sup0 = arch(x_arch)
        initial_loss = torch.mean((low0 - target)**2 + (sup0 - target)**2)

    for epoch in range(50):
        optimizer_arch.zero_grad()
        low, sup = arch(x_arch)
        loss_arch = torch.mean((low - target)**2 + (sup - target)**2)
        loss_arch.backward()
        optimizer_arch.step()

    with torch.no_grad():
        final_low, final_sup = arch(x_arch)
        final_loss = torch.mean((final_low - target)**2 + (final_sup - target)**2)

    print(f"Shapes: low {tuple(final_low.shape)}, sup {tuple(final_sup.shape)}")
    print(f"Initial loss: {initial_loss.item():.4f}")
    print(f"Final loss:   {final_loss.item():.4f}")
    print(f"Low  (mean/min/max): {final_low.mean():.4f} / {final_low.min():.4f} / {final_low.max():.4f}")
    print(f"Sup  (mean/min/max): {final_sup.mean():.4f} / {final_sup.min():.4f} / {final_sup.max():.4f}")

    assert final_low.shape == (B, 1) and final_sup.shape == (B, 1)
    assert (final_low <= final_sup).all(), "MinMax should give low <= sup"
    assert final_loss.item() < initial_loss.item(), "loss did not decrease"

    with torch.no_grad():
        for b in range(B):
            lo_b, up_b = arch(x_arch[b])
            assert torch.allclose(final_low[b], lo_b, atol=1e-5)
            assert torch.allclose(final_sup[b], up_b, atol=1e-5)

    print("Architecture Test (batch): SUCCESS")


    phi = [-5,-2, 1, 4]
    x = torch.Tensor([-1, 1,-3, 2]).unsqueeze(0)

    test = NeuroneMacSum(4,1)
    test.parametres = nn.Parameter(torch.Tensor(phi).unsqueeze(0))

    with torch.no_grad():
        print(test(x))
>>>>>>> 454aeec7d31b6b79317fee17a0764b647b69480d
