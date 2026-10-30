import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from src.training.point_estimator import MeanEstim, Disjunctive, Conjunctive


POINT_ESTIM_FUNCTION = {
    "mean": MeanEstim,
    "disjunctive": Disjunctive,
    "conjunctive": Conjunctive,
}

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

class NeuroneMacSumFunction(torch.autograd.Function):
    @staticmethod
    def _diff(c: Tensor) -> Tensor:
        zeros = torch.zeros_like(c[..., :1])
        return c - torch.cat([zeros, c[..., :-1]], dim=-1)

    @staticmethod
    def forward(ctx, x, w):
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        w_plus, idx_d = torch.sort(w.clamp(min=0), dim=1, descending=True)
        w_moin, idx_c = torch.sort(w.clamp(max=0), dim=1, descending=False)

        xp, xm = x.clamp(min=0), x.clamp(max=0)

        xp_d, xm_d = xp[:, idx_d], xm[:, idx_d]
        xp_c, xm_c = xp[:, idx_c], xm[:, idx_c]

        beta_sup = torch.cummax(xp_d, -1).values + torch.cummax(xm_d, -1).values
        gamma_sup = torch.cummin(xp_c, -1).values + torch.cummin(xm_c, -1).values

        beta_low = torch.cummin(xp_d, -1).values + torch.cummin(xm_d, -1).values
        gamma_low = torch.cummax(xp_c, -1).values + torch.cummax(xm_c, -1).values

        diff_beta_sup = NeuroneMacSumFunction._diff(beta_sup)
        diff_gamma_sup = NeuroneMacSumFunction._diff(gamma_sup)
        diff_beta_low = NeuroneMacSumFunction._diff(beta_low)
        diff_gamma_low = NeuroneMacSumFunction._diff(gamma_low)

        y_sup = (w_plus * diff_beta_sup).sum(-1) + (w_moin * diff_gamma_sup).sum(-1)
        y_low = (w_plus * diff_beta_low).sum(-1) + (w_moin * diff_gamma_low).sum(-1)

        ctx.save_for_backward(x, w, diff_beta_sup, diff_gamma_sup, diff_beta_low, diff_gamma_low, idx_d, idx_c)
        ctx.squeeze = squeeze

        if squeeze:
            y_low, y_sup = y_low.squeeze(0), y_sup.squeeze(0)
        return y_low, y_sup

    @staticmethod
    def backward(ctx, grad_y_low, grad_y_sup):
        x, w, diff_beta_sup, diff_gamma_sup, diff_beta_low, diff_gamma_low, idx_d, idx_c = ctx.saved_tensors

        if ctx.squeeze:
            grad_y_low = grad_y_low.unsqueeze(0)
            grad_y_sup = grad_y_sup.unsqueeze(0)

        mask_pos = (w >= 0).unsqueeze(0)

        diff_beta_sup_orig = torch.gather(diff_beta_sup, -1, idx_d.unsqueeze(0).expand(x.size(0), -1, -1))
        diff_gamma_sup_orig = torch.gather(diff_gamma_sup, -1, idx_c.unsqueeze(0).expand(x.size(0), -1, -1))

        diff_beta_low_orig = torch.gather(diff_beta_low, -1, idx_d.unsqueeze(0).expand(x.size(0), -1, -1))
        diff_gamma_low_orig = torch.gather(diff_gamma_low, -1, idx_c.unsqueeze(0).expand(x.size(0), -1, -1))

        delta_sup = torch.where(mask_pos, diff_beta_sup_orig, diff_gamma_sup_orig)
        delta_low = torch.where(mask_pos, diff_beta_low_orig, diff_gamma_low_orig)

        grad_w = (grad_y_sup.unsqueeze(-1) * delta_sup + grad_y_low.unsqueeze(-1) * delta_low).sum(dim=0)

        w_expanded = w.unsqueeze(0)
        grad_x_sup = (grad_y_sup.unsqueeze(-1) * w_expanded).sum(dim=1)  
        grad_x_low = (grad_y_low.unsqueeze(-1) * w_expanded).sum(dim=1)  
        mask_sup_active = (grad_y_sup != 0).any(dim=1, keepdim=True)   
        grad_x = torch.where(mask_sup_active, grad_x_sup, grad_x_low)

        if ctx.squeeze:
            grad_x = grad_x.squeeze(0)

        return grad_x, grad_w


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
        return NeuroneMacSumFunction.apply(x, self.parametres)

    @staticmethod
    def backward(ctx, grad_y_low, grad_y_sup):
        x, w, diff_beta_sup, diff_gamma_sup, diff_beta_low, diff_gamma_low, idx_d, idx_c = ctx.saved_tensors

        if ctx.squeeze:
            grad_y_low = grad_y_low.unsqueeze(0)
            grad_y_sup = grad_y_sup.unsqueeze(0)

        mask_pos = (w >= 0).unsqueeze(0)

        diff_beta_sup_orig = torch.gather(diff_beta_sup, -1, idx_d.unsqueeze(0).expand(x.size(0), -1, -1))
        diff_gamma_sup_orig = torch.gather(diff_gamma_sup, -1, idx_c.unsqueeze(0).expand(x.size(0), -1, -1))

        diff_beta_low_orig = torch.gather(diff_beta_low, -1, idx_d.unsqueeze(0).expand(x.size(0), -1, -1))
        diff_gamma_low_orig = torch.gather(diff_gamma_low, -1, idx_c.unsqueeze(0).expand(x.size(0), -1, -1))

        delta_sup = torch.where(mask_pos, diff_beta_sup_orig, diff_gamma_sup_orig)
        delta_low = torch.where(mask_pos, diff_beta_low_orig, diff_gamma_low_orig)

        grad_w = (grad_y_sup.unsqueeze(-1) * delta_sup + grad_y_low.unsqueeze(-1) * delta_low).sum(dim=0)

        w_expanded = w.unsqueeze(0)
        grad_x_sup = (grad_y_sup.unsqueeze(-1) * w_expanded).sum(dim=1)
        grad_x_low = (grad_y_low.unsqueeze(-1) * w_expanded).sum(dim=1)

        mask_sup_active = (grad_y_sup != 0).unsqueeze(-1)
        grad_x = torch.where(mask_sup_active, grad_x_sup, grad_x_low)

        if ctx.squeeze:
            grad_x = grad_x.squeeze(0)

        return grad_x, grad_w

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
        return NeuroneMacSumFunction.apply(x, self.parametres)

class MinMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.minimum(a, b), torch.maximum(a, b)


class MacSumDenseMinMax(nn.Module):
    def __init__(
        self,
        MacSumIn: int,
        MacSumOut: int,
        output_size: int,
        linearStart: int,
        linearN: int,
        increment: int,
        activation_function: str = "relu",
        aggregation_type: str = "mean",
    ):
        super().__init__()
        self.MacSumIn = MacSumIn
        self.MacSumOut = MacSumOut
        self.linearStart = linearStart
        self.linearN = linearN
        self.increment = increment
        self.output_size = output_size
        self.aggregation_type = aggregation_type

        last_size = MacSumOut - increment * linearN * (linearN + 1) // 2

        assert activation_function in ACTIVATION_FUNCTION_REGISTERY, "activation function not in registery"
        assert aggregation_type in POINT_ESTIM_FUNCTION, "aggregation_type not in registery"
        assert linearN >= 0, "linearN can't be negative"
        assert increment >= 0, "increment can't be negative"
        assert linearStart <= MacSumOut, "linearStart can't be bigger than MacSumOut"
        assert output_size >= 1, "output_size must be positive"
        assert output_size <= last_size, "output size can't be bigger than the last linear layer input size"

        self.MacSum = NeuroneMacSum(self.MacSumIn, self.MacSumOut)
        self.MinMax = MinMax()
        self.activation_function = ACTIVATION_FUNCTION_REGISTERY[activation_function]

        self.aggregator = POINT_ESTIM_FUNCTION[aggregation_type]()

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
        if isinstance(x, (tuple, list)):
            x_low, x_up = x
            if self.aggregation_type in ["disjunctive", "conjunctive"]:
                x_agg = self.aggregator((x_low, x_up), self.MacSum.parametres)
            else:
                x_agg = self.aggregator((x_low, x_up))
            
            skip_input = x_agg
        else:
            x_agg = x
            skip_input = x

        
        y_low, y_sup = self.MacSum(x_agg)
        skip = self.skip(skip_input)
        low, sup = self.MinMax(self.ForwardPass(y_low), self.ForwardPass(y_sup))

        return low + skip, sup + skip


class ArchitectureMacSum(nn.Module):
    def __init__(
        self,
        input_shape,
        lis_MacSumOut,
        lis_output_size,
        lis_activation,
        lis_increments,
        lis_linears_start,
        lis_LinearN,
        lis_point_estim_function
    ):
        super().__init__()

        assert all(i in ACTIVATION_FUNCTION_REGISTERY for i in lis_activation), "one of the activation function is not in the registery"
        assert all(i in POINT_ESTIM_FUNCTION for i in lis_point_estim_function), "one of the point estimation function is not in the registery"
        assert (
            len(lis_MacSumOut)
            == len(lis_output_size)
            == len(lis_activation)
            == len(lis_increments)
            == len(lis_linears_start)
            == len(lis_point_estim_function)
            == len(lis_LinearN)
        ), "length does not match"

        self.input_shape = input_shape
        current_input_shape = input_shape
        

        self.blocks = nn.ModuleList()
        for macsumout, output_size, activation, increments, linears_start, point_estim, linearN in zip(
            lis_MacSumOut,
            lis_output_size,
            lis_activation,
            lis_increments,
            lis_linears_start,
            lis_point_estim_function,
            lis_LinearN,
        ):
            self.blocks.append(
                MacSumDenseMinMax(
                    current_input_shape,
                    macsumout,
                    output_size,
                    linears_start,
                    linearN,
                    increments,
                    activation,
                    aggregation_type=point_estim,
                )
            )
            current_input_shape = output_size

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out
    
