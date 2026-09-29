"""Small mathematical/API checks; no training or experiment reruns."""
import torch
from torch import nn

from count_fm import sample_xt, sample_rt, model_forward, model_loss, sample_euler
from count_fm.coupling import _get_ot_cost_matrix, _soft_ot_pair_batch
from count_fm.models import MLP, MLP_rate, Adapted_FlatConvUNet, Adapted_FlatConvUNet_rate
from count_fm.models.pcx import CondCountFMNet, CondCountFMNet_MLP
from count_fm.sampling import jump_probabilities


def test_bridge_endpoints_and_support():
    x0 = torch.tensor([[0, 7, 3], [8, 1, 0]])
    x1 = torch.tensor([[5, 2, 3], [0, 6, 0]])
    assert torch.equal(sample_xt(x0, x1, torch.zeros(2, 1)), x0)
    assert torch.equal(sample_xt(x0, x1, torch.ones(2, 1)), x1)
    xt = sample_xt(x0, x1, torch.full((2, 1), 0.4))
    assert ((xt >= torch.minimum(x0, x1)) & (xt <= torch.maximum(x0, x1))).all()


def test_exact_bridge_rates_and_zero_death_boundary():
    xt = torch.tensor([[2., 5., 0.]])
    x1 = torch.tensor([[6., 1., 0.]])
    rates, zero = sample_rt(xt, x1, torch.tensor([[0.5]]), eps_t=0.1)
    torch.testing.assert_close(rates, torch.tensor([[8., 0., 0., 0., 8., 0.]]))
    net = MLP_rate(MLP(3, out_dim=6, w=8, time_varying=True))
    pred = model_forward(torch.cat([xt, torch.zeros(1, 1)], 1), False, net, 3, zero)
    assert pred[0, 5] == 0
    loss = model_loss("poisson", pred, rates)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_jump_probabilities_match_competing_hazards():
    birth = torch.tensor([0., 2., 1e-12], dtype=torch.float64)
    death = torch.tensor([0., 3., 0.], dtype=torch.float64)
    stay, up, down = jump_probabilities(birth, death, 0.2)
    torch.testing.assert_close(stay + up + down, torch.ones_like(stay))
    assert (stay[0], up[0], down[0]) == (1., 0., 0.)
    torch.testing.assert_close(stay[1], torch.exp(torch.tensor(-1., dtype=torch.float64)))
    torch.testing.assert_close(up[1] / down[1], torch.tensor(2 / 3, dtype=torch.float64))
    assert up[2] > 0 and down[2] == 0


class ConstantRates(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.times = []

    def forward(self, z):
        assert not self.training and not torch.is_grad_enabled()
        self.times.append(z[:, -1].clone())
        return self.values.expand(z.shape[0], -1)


def test_sampler_grid_boundary_and_mode_restore():
    net = ConstantRates([0., 0., 100., 100.])
    x0 = torch.zeros(3, 2)
    result, path = sample_euler(net, 4, x0, "cpu", eps_t=0.2)
    assert net.training and not path.requires_grad
    assert torch.equal(result, x0.long()) and path.shape == (5, 3, 2)
    torch.testing.assert_close(torch.stack(net.times)[:, 0], torch.tensor([0., .2, .4, .6]))
    birth, death = ConstantRates([0., 0.]), ConstantRates([10., 10.])
    result, _ = sample_euler([birth, death], 2, x0, "cpu", separate_heads=True)
    assert not result.any() and birth.training and death.training


def test_pcx_dropout_removes_all_covariates():
    models = [
        CondCountFMNet_MLP(3, 1, [3], emb_dim=4, hidden=8, depth=1),
        CondCountFMNet(3, 1, [3], emb_dim=4, cond_dim=4, d_model=8,
                       depth=1, n_heads=2, chunk_size=4),
    ]
    for net in models:
        net.eval()
        xt, t = torch.zeros(2, 3), torch.full((2, 1), .4)
        cont, cat = torch.tensor([[9.], [-7.]]), torch.tensor([[0], [2]])
        masked = net(xt, t, cont, cat, is_uncond_mask=torch.ones(2, dtype=torch.bool))
        unconditional = net.forward_uncond(xt, t)
        for actual, expected in zip(masked, unconditional):
            torch.testing.assert_close(actual, expected)


def test_unet_wrapper_forwards_covariates():
    base = Adapted_FlatConvUNet(3, out_dim=6, time_varying=True, cov_dim=2,
                               channels=(4, 8, 8), embed_dim=8)
    net = Adapted_FlatConvUNet_rate(base)
    z, c = torch.zeros(2, 4), torch.ones(2, 2)
    expected = base(z, c).clamp(max=net.log_cap).exp() + net.eps
    torch.testing.assert_close(net(z, c), expected)


def test_ot_cost_and_unequal_independent_batches():
    x0, x1 = torch.tensor([[0, 2], [3, 1]]), torch.tensor([[1, 0], [3, 1], [4, 5]])
    cost = _get_ot_cost_matrix(x0, x1, "sym_poisson")
    a, b = x0[:, None].float(), x1[None].float()
    expected = ((a - b) * ((a + 1e-8).log() - (b + 1e-8).log())).sum(-1)
    torch.testing.assert_close(cost, expected)
    assert _get_ot_cost_matrix(x0, x1, "l2").shape == (2, 3)
    paired0, paired1 = _soft_ot_pair_batch(x0, x1, "none", out_size=3)
    assert paired0.shape == paired1.shape == (3, 2)
