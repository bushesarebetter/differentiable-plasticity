import numpy as np
import torch

import plasticity as P


def test_alpha_zero_matches_non_plastic():
    torch.manual_seed(0)
    plastic, fixed = P.RNN(n=20, plastic=True), P.RNN(n=20, plastic=False)
    with torch.no_grad():
        plastic.alpha.zero_()
        plastic.eta.fill_(0.5)  # big eta so Hebb is far from zero; must still have no effect
        fixed.w.copy_(plastic.w)
    inputs = torch.randn(30, 4, 20)
    assert torch.allclose(plastic(inputs), fixed(inputs), atol=1e-6)


def test_hebb_update_matches_formula():
    rng = np.random.default_rng(0)
    hebb, pre, post = rng.normal(size=(2, 5, 5)), rng.normal(size=(2, 5)), rng.normal(size=(2, 5))
    eta = 0.3
    expected = np.empty_like(hebb)
    for b in range(2):
        for i in range(5):
            for j in range(5):
                expected[b, i, j] = (1 - eta) * hebb[b, i, j] + eta * pre[b, i] * post[b, j]
    got = P.hebb_update(torch.tensor(hebb), torch.tensor(pre), torch.tensor(post), eta)
    assert np.allclose(got.numpy(), expected)


def test_step_uses_hebb_trace():
    # one step by hand: y_new = tanh(y @ (w + alpha * hebb) + inp), trace updated from (y, y_new)
    torch.manual_seed(1)
    net = P.RNN(n=6)
    y, inp, hebb = torch.randn(3, 6), torch.randn(3, 6), torch.randn(3, 6, 6)
    with torch.no_grad():
        y_new, hebb_new = net.step(inp, y, hebb)
        for b in range(3):
            ref = torch.tanh(y[b] @ (net.w + net.alpha * hebb[b]) + inp[b])
            assert torch.allclose(y_new[b], ref, atol=1e-6)
            ref_h = (1 - net.eta) * hebb[b] + net.eta * torch.outer(y[b], ref)
            assert torch.allclose(hebb_new[b], ref_h, atol=1e-6)


def test_task_shapes_and_probe():
    inputs, target, mask, pats, probe = P.make_batch(5)
    T = P.CYCLES * P.N_PATTERNS * (P.PRES_TIME + P.DELAY) + P.PRES_TIME
    assert inputs.shape == (T, 5, P.N)
    assert set(target.unique().tolist()) <= {-1.0, 1.0}
    assert torch.all(probe[mask] == 0) and torch.equal(probe[~mask], target[~mask])
    # target is one of the stored patterns
    assert all((pats[b] == target[b]).all(1).any() for b in range(5))
