import torch 
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F

class NeuroneMacSum(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.parametres = nn.Parameter(torch.rand(self.n_out, self.n_in))   

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:

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

                beta_sup = x_k_decroissante_plus.max() + x_k_decroissante_moin.max()
                beta_low = (-x_k_decroissante_plus).max() + (-x_k_decroissante_moin).max()
                gamma_sup = x_k_croissante_plus.min() + x_k_croissante_moin.min()
                gamma_low = (-x_k_croissante_plus).min() + (-x_k_croissante_moin).min()

                y_sup += difference_plus*beta_sup + difference_moin*gamma_sup
                y_low += -(difference_plus*beta_low + difference_moin*gamma_low)

            liste_y_low.append(y_low)
            liste_y_sup.append(y_sup)

        return torch.stack(liste_y_low), torch.stack(liste_y_sup)  

class MacSumArchitecture(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        pass
            








if __name__ == "__main__": 

    torch.manual_seed(0)

    n_in = 5
    n_out = 3   # plus parlant que 1 pour vérifier les formes
    neurone = NeuroneMacSum(n_in, n_out)
    x = torch.tensor([0.5, -1.0, 2.0, -0.3, 1.5])

    y_low, y_sup = neurone(x)
    print("y_low :", y_low)
    print("y_sup :", y_sup)

    # 2. Formes : un scalaire par neurone de sortie
    assert y_low.shape == (n_out,) and y_sup.shape == (n_out,)

    # 3. Cohérence : la borne basse doit être <= la borne haute (pour chaque sortie)
    print("y_low <= y_sup :", (y_low <= y_sup).all().item())

    # 4. Backward : .sum() car loss doit être un scalaire
    loss = (y_sup - y_low).sum()
    loss.backward()
    print("grad :", neurone.parametres.grad)
    assert neurone.parametres.grad is not None
    assert neurone.parametres.grad.shape == (n_out, n_in)

    # 5. Mini entraînement
    opt = torch.optim.Adam(neurone.parameters(), lr=0.05)
    for epoch in range(50):
        opt.zero_grad()
        y_low, y_sup = neurone(x)
        loss = ((y_sup - 1.0) ** 2 + (y_low + 1.0) ** 2).sum()
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch:2d} | loss {loss.item():.4f}")