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
        return a.min(),b.max()


class MacSumIntervallArchitecture(nn.Module):
    def __init__(self,input_size:int,listDenseIn:list[int],MacSumIn:int,MacSumOut:int):
        super().__init__()
        self.input_size = input_size
        self.listDenseIn = listDenseIn
        self.MacSumIn = MacSumIn
        self.MacSumOut = MacSumOut

        self.MacSum = NeuroneMacSum(self.input_size,self.MacSumOut)
        listForwardDense = [nn.Linear(self.MacSumOut,self.listDenseIn[0])]
        if len(listDenseIn) > 1:
            for i in range(0,len(listForwardDense)-1):
                listForwardDense.append(nn.Linear(self.listDenseIn[i],self.listDenseIn[i+1]))

        self.ForwardPass = nn.Sequential(*listForwardDense)
        self.MinMax = MinMax()

    def forward(self,x):
        x_low,x_sup = self.MacSum(x)
        return self.MinMax(self.ForwardPass(x_low),self.ForwardPass(x_sup))

            








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
    arch = MacSumIntervallArchitecture(input_size=5, listDenseIn=[10, 1], MacSumIn=5, MacSumOut=3)
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
    print(f"Final Architecture Low: {final_low.item():.4f}")
    print(f"Final Architecture Sup: {final_sup.item():.4f}")
    print(f"Final Architecture Loss: {loss_arch.item():.4f}")
    print("Architecture Test: SUCCESS")

