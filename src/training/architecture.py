import torch 
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F

class NeuroneMacSum(nn.Module):
    def __init__(self, n_in:int, n_out:int):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.parametres = nn.Parameter(torch.rand(self.n_out, self.n_in))   

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Pseudo Code 

        x_decroissante, _ = torch.sort(x, descending=True)
        x_croissante, _ = torch.sort(x, descending=False)
        liste_y_low = []
        liste_y_sup = []

        for j in range(self.n_out):
            _, i_parameters_decroissante = torch.sort(self.parametres[j], descending=True)
            _, i_parameters_croissante = torch.sort(self.parametres[j], descending=False)
            y_sup = 0
            y_low = 0

            for k in range(0, self.n_in):
                difference_plus = torch.clamp(self.parametres[j, i_parameters_decroissante[k]], min=0) \
                    - (torch.clamp(self.parametres[j, i_parameters_decroissante[k+1]], min=0) if k + 1 < self.n_in else 0)
                difference_moin = torch.clamp(self.parametres[j, i_parameters_croissante[k]], max=0) \
                    - (torch.clamp(self.parametres[j, i_parameters_croissante[k+1]], max=0) if k + 1 < self.n_in else 0)

                x_k_decroissante = x_decroissante[:k+1]
                x_k_croissante = x_croissante[:k+1]

                x_k_decroissante_plus = torch.clamp(x_k_decroissante, min=0)
                x_k_decroissante_moin = torch.clamp(x_k_decroissante, max=0)
                x_k_croissante_plus = torch.clamp(x_k_croissante, min=0)
                x_k_croissante_moin = torch.clamp(x_k_croissante, max=0)

                beta_k_sup = x_k_decroissante_plus.max() + x_k_decroissante_moin.max()
                beta_k_low = (-x_k_decroissante_plus).max() + (-x_k_decroissante_moin).max()
                gamma_k_sup = x_k_croissante_plus.min() + x_k_croissante_moin.min()
                gamma_k_low = (-x_k_croissante_plus).min() + (-x_k_croissante_moin).min()

                y_sup += difference_plus*beta_k_sup + difference_moin*gamma_k_sup
                y_low += -(difference_plus*beta_k_low + difference_moin*gamma_k_low)

            liste_y_low.append(y_low)
            liste_y_sup.append(y_sup)

        return torch.stack(liste_y_low), torch.stack(liste_y_sup)  
        """
        
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)                                 

        w = self.parametres                                      

      
        w_dec, _ = torch.sort(w, dim=1, descending=True)
        w_cro, _ = torch.sort(w, dim=1, descending=False)
        w_plus = torch.clamp(w_dec, min=0)
        w_moin = torch.clamp(w_cro, max=0)

       
        xd, _ = torch.sort(x, dim=1, descending=True)
        xc, _ = torch.sort(x, dim=1, descending=False)

        xd_p, xd_m = torch.clamp(xd, min=0), torch.clamp(xd, max=0)
        xc_p, xc_m = torch.clamp(xc, min=0), torch.clamp(xc, max=0)

        beta_k_sup  = torch.cummax(xd_p, dim=-1).values  + torch.cummax(xd_m, dim=-1).values
        beta_k_low  = torch.cummax(-xd_p, dim=-1).values + torch.cummax(-xd_m, dim=-1).values
        gamma_k_sup = torch.cummin(xc_p, dim=-1).values  + torch.cummin(xc_m, dim=-1).values
        gamma_k_low = torch.cummin(-xc_p, dim=-1).values + torch.cummin(-xc_m, dim=-1).values

      
        zeros = torch.zeros_like(beta_k_sup[..., :1])
        beta_k_1_sup  = torch.cat([zeros, beta_k_sup[..., :-1]],  dim=-1)
        beta_k_1_low  = torch.cat([zeros, beta_k_low[..., :-1]],  dim=-1)
        gamma_k_1_sup = torch.cat([zeros, gamma_k_sup[..., :-1]], dim=-1)
        gamma_k_1_low = torch.cat([zeros, gamma_k_low[..., :-1]], dim=-1)

        
        y_sup = (beta_k_sup - beta_k_1_sup) @ w_plus.T + (gamma_k_sup - gamma_k_1_sup) @ w_moin.T
        y_low = -((beta_k_low - beta_k_1_low) @ w_plus.T + (gamma_k_low - gamma_k_1_low) @ w_moin.T)

        if squeeze:
            y_low, y_sup = y_low.squeeze(0), y_sup.squeeze(0)
        return y_low, y_sup


class MinMax(nn.Module): 
    def __init__(self):
        super().__init__()

    def forward(self,a,b): 
        return torch.minimum(a, b), torch.maximum(a, b)

class MacSumDenseMinMax(nn.Module):
    def __init__(self, MacSumIn: int, MacSumOut: int, output_size: int, linearStart: int,
                 linearN: int, increment: int, inproper=False):
        super().__init__()
        self.MacSumIn = MacSumIn
        self.MacSumOut = MacSumOut

        self.linearStart = linearStart
        self.linearN = linearN
        self.increment = increment
        self.inproper = inproper
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
        if self.inproper:
            x_low, x_sup = x_sup, x_low

        return self.MinMax(self.ForwardPass(x_low), self.ForwardPass(x_sup))        



            








if __name__ == "__main__": 
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

    print("\n--- TEST ARCHITECTURE ---")
    arch = MacSumDenseMinMax(MacSumIn=5, MacSumOut=8, output_size=1,
                            linearStart=8, linearN=2, increment=1, inproper=False)
    x_arch = torch.randn(5)

    optimizer_arch = torch.optim.Adam(arch.parameters(), lr=0.01)
    target = torch.tensor([1.0])

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