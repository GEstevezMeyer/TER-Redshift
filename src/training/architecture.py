import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from src.training.point_estimator import MeanEstim


POINT_ESTIM_FUNCTION = {"mean": MeanEstim}

ACTIVATION_FUNCTION_REGISTERY = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "softplus": nn.Softplus,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "hardswish": nn.Hardswish,
    "identity": nn.Identity,
}


class NeuroneMacSum(nn.Module):
    @staticmethod
    def _diff(c: Tensor) -> Tensor:
        zeros = torch.zeros_like(c[..., :1])
        return c - torch.cat([zeros, c[..., :-1]], dim=-1)

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        std = 1.0 / (n_in ** 0.5)
        self.parametres = nn.Parameter(torch.empty(n_out, n_in).normal_(0.0, std))

    def forward(self, x):
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        w = self.parametres

        w_plus, idx_d = torch.sort(w.clamp(min=0), dim=1, descending=True)
        w_moin, idx_c = torch.sort(w.clamp(max=0), dim=1, descending=False)

        xp, xm = x.clamp(min=0), x.clamp(max=0)

        xp_d, xm_d = xp[:, idx_d], xm[:, idx_d]
        xp_c, xm_c = xp[:, idx_c], xm[:, idx_c]

        beta_sup = torch.cummax(xp_d, -1).values + torch.cummax(xm_d, -1).values
        gamma_sup = torch.cummin(xp_c, -1).values + torch.cummin(xm_c, -1).values

        beta_low = torch.cummin(xp_d, -1).values + torch.cummin(xm_d, -1).values
        gamma_low = torch.cummax(xp_c, -1).values + torch.cummax(xm_c, -1).values

        y_sup = (w_plus * self._diff(beta_sup)).sum(-1) + (w_moin * self._diff(gamma_sup)).sum(-1)
        y_low = (w_plus * self._diff(beta_low)).sum(-1) + (w_moin * self._diff(gamma_low)).sum(-1)

        if squeeze:
            y_low, y_sup = y_low.squeeze(0), y_sup.squeeze(0)
        return y_low, y_sup


class MinMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.minimum(a, b), torch.maximum(a, b)


class MacSumDenseMinMax(nn.Module):
    def __init__(self, MacSumIn: int, MacSumOut: int, output_size: int, linearStart: int,
                 linearN: int, increment: int, activation_function: str = "relu"):
        super().__init__()
        self.MacSumIn = MacSumIn
        self.MacSumOut = MacSumOut
        self.linearStart = linearStart
        self.linearN = linearN
        self.increment = increment
        self.output_size = output_size

        last_size = MacSumOut - increment * linearN * (linearN + 1) // 2

        assert activation_function in ACTIVATION_FUNCTION_REGISTERY, "activation function not in registery"
        assert linearN >= 0, "linearN can't be negative"
        assert increment >= 0, "increment can't be negative"
        assert linearStart <= MacSumOut, "linearStart can't be bigger than MacSumOut"
        assert output_size >= 1, "output_size must be positive"
        assert output_size <= last_size, "output size can't be bigger than the last linear layer input size"

        self.MacSum = NeuroneMacSum(self.MacSumIn, self.MacSumOut)
        self.MinMax = MinMax()
        self.activation_function = ACTIVATION_FUNCTION_REGISTERY[activation_function]

        listLinear = []
        current_input = self.MacSumOut

        for i in range(0, linearN):
            listLinear.append(nn.Linear(current_input, current_input - (i + 1) * increment))
            listLinear.append(self.activation_function())
            current_input = current_input - (i + 1) * increment

        listLinear.append(nn.Linear(current_input, output_size))

        self.ForwardPass = nn.Sequential(*listLinear)
        self.skip = nn.Linear(MacSumIn, output_size)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.ForwardPass.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)
        nn.init.kaiming_normal_(self.skip.weight, nonlinearity="linear")
        nn.init.zeros_(self.skip.bias)

    def forward(self, x):
        x_low, x_sup = self.MacSum(x)
        skip = self.skip(x)
        low, sup = self.MinMax(self.ForwardPass(x_low), self.ForwardPass(x_sup))
        return low + skip, sup + skip


class ArchitectureMacSum(nn.Module):
    def __init__(self, input_shape, lis_MacSumOut, lis_output_size, lis_activation,
                 lis_increments, lis_linears_start, lis_LinearN, lis_point_estim_function):
        super().__init__()

        assert all(i in ACTIVATION_FUNCTION_REGISTERY for i in lis_activation), "one of the activation function is not in the registery"
        assert all(i in POINT_ESTIM_FUNCTION for i in lis_point_estim_function), "one of the point estimation function is not in the registery"
        assert (len(lis_MacSumOut) == len(lis_output_size) == len(lis_activation) == len(lis_increments)
                == len(lis_linears_start) == len(lis_point_estim_function) == len(lis_LinearN)), "lenght are not matchable"

        self.input_shape = input_shape
        current_input_shape = input_shape
        blocks = []
        n = len(lis_MacSumOut)
        for i, (macsumout, output_size, activation, increments,
                linears_start, point_estim, linearN) in enumerate(zip(
                    lis_MacSumOut, lis_output_size, lis_activation, lis_increments,
                    lis_linears_start, lis_point_estim_function, lis_LinearN)):

            blocks.append(MacSumDenseMinMax(current_input_shape, macsumout, output_size,
                                            linears_start, linearN, increments,
                                            activation))
            if i < n - 1:
                blocks.append(POINT_ESTIM_FUNCTION[point_estim]())
            current_input_shape = output_size

        self.ForwardPass = nn.Sequential(*blocks)

    def forward(self, x):
        return self.ForwardPass(x)


if __name__ == "__main__":
    pass
