import torch
import torch.nn as nn

from src.training.architecture import NeuroneMacSum, MacSumDenseMinMax


def test_neurone():
    torch.manual_seed(0)
    n_in, n_out = 5, 3
    neurone = NeuroneMacSum(n_in, n_out)
    x = torch.tensor([0.5, -1.0, 2.0, -0.3, 1.5])
    y_low, y_sup = neurone(x)

    opt = torch.optim.Adam(neurone.parameters(), lr=0.05)
    for epoch in range(50):
        opt.zero_grad()
        yl, ys = neurone(x)
        loss = ((ys - 1.0) ** 2 + (yl + 1.0) ** 2).sum()
        loss.backward()
        opt.step()
    assert loss.item() < 10.0


def test_architecture():
    torch.manual_seed(0)
    arch = MacSumDenseMinMax(MacSumIn=5, MacSumOut=8, output_size=1,
                             linearStart=8, linearN=2, increment=1)
    x_arch = torch.randn(5)

    optimizer_arch = torch.optim.Adam(arch.parameters(), lr=0.01)
    target = torch.tensor([1.0])

    for epoch in range(50):
        optimizer_arch.zero_grad()
        low, sup = arch(x_arch)
        loss_arch = torch.mean((low - target) ** 2 + (sup - target) ** 2)
        loss_arch.backward()
        optimizer_arch.step()

    final_low, final_sup = arch(x_arch)
    final_loss = torch.mean((final_low - target) ** 2 + (final_sup - target) ** 2)

    assert final_low.numel() == 1 and final_sup.numel() == 1
    assert (final_low <= final_sup).all()
    assert final_loss.item() < 1.0


def test_architecture_batch():
    torch.manual_seed(0)
    B = 32
    arch = MacSumDenseMinMax(MacSumIn=5, MacSumOut=8, output_size=1,
                             linearStart=8, linearN=2, increment=1)
    x_arch = torch.randn(B, 5)
    target = torch.ones(B, 1)

    optimizer_arch = torch.optim.Adam(arch.parameters(), lr=0.01)

    with torch.no_grad():
        low0, sup0 = arch(x_arch)
        initial_loss = torch.mean((low0 - target) ** 2 + (sup0 - target) ** 2)

    for epoch in range(50):
        optimizer_arch.zero_grad()
        low, sup = arch(x_arch)
        loss_arch = torch.mean((low - target) ** 2 + (sup - target) ** 2)
        loss_arch.backward()
        optimizer_arch.step()

    with torch.no_grad():
        final_low, final_sup = arch(x_arch)
        final_loss = torch.mean((final_low - target) ** 2 + (final_sup - target) ** 2)

    assert final_low.shape == (B, 1) and final_sup.shape == (B, 1)
    assert (final_low <= final_sup).all()
    assert final_loss.item() < initial_loss.item()

    for b in range(B):
        lo_b, up_b = arch(x_arch[b])
        assert torch.allclose(final_low[b], lo_b, atol=1e-5)
        assert torch.allclose(final_sup[b], up_b, atol=1e-5)


def test_phi():
    phi = [-5, -2, 1, 4]
    x = torch.Tensor([-1, 1, -3, 2]).unsqueeze(0)

    test = NeuroneMacSum(4, 1)
    test.parametres = nn.Parameter(torch.Tensor(phi).unsqueeze(0))

    with torch.no_grad():
        y_low, y_sup = test(x)
        assert y_low.shape == y_sup.shape
        assert (y_low <= y_sup + 1e-6).all()
        assert True

