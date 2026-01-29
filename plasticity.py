import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

N = 50          # pattern size = number of neurons
N_PATTERNS = 3
PRES_TIME = 6   # steps each pattern is shown
DELAY = 4       # blank steps between patterns
CYCLES = 2      # how many times the whole set is shown
DEGRADE = 0.5   # fraction of bits zeroed in the probe


class RNN(nn.Module):
    """x(t) = tanh(x(t-1) @ (w + alpha * Hebb) + input), with Hebb reset every episode."""

    def __init__(self, n=N, plastic=True):
        super().__init__()
        self.plastic = plastic
        self.w = nn.Parameter(0.01 * torch.randn(n, n))
        if plastic:
            self.alpha = nn.Parameter(0.01 * torch.randn(n, n))
            self.eta = nn.Parameter(torch.tensor(0.01))

    def step(self, inp, y, hebb):
        if not self.plastic:
            return torch.tanh(y @ self.w + inp), hebb
        w_eff = self.w + self.alpha * hebb                       # (B, n, n), row = presynaptic
        y_new = torch.tanh(torch.bmm(y.unsqueeze(1), w_eff).squeeze(1) + inp)
        hebb = hebb_update(hebb, y, y_new, self.eta)
        return y_new, hebb

    def forward(self, inputs):
        T, B, n = inputs.shape
        y = torch.zeros(B, n)
        hebb = torch.zeros(B, n, n) if self.plastic else None
        for t in range(T):
            y, hebb = self.step(inputs[t], y, hebb)
        return y


def hebb_update(hebb, pre, post, eta):
    # Hebb_ij <- (1 - eta) Hebb_ij + eta * x_i(t-1) x_j(t)
    return (1 - eta) * hebb + eta * pre.unsqueeze(2) * post.unsqueeze(1)


def make_batch(batch, n=N):
    pats = torch.randint(0, 2, (batch, N_PATTERNS, n)).float() * 2 - 1
    blank = torch.zeros(batch, n)
    seq = []
    for _ in range(CYCLES):
        for p in torch.randperm(N_PATTERNS):
            seq += [pats[:, p]] * PRES_TIME + [blank] * DELAY
    target = pats[torch.arange(batch), torch.randint(0, N_PATTERNS, (batch,))]
    mask = torch.rand(batch, n) < DEGRADE
    probe = target.masked_fill(mask, 0.0)
    seq += [probe] * PRES_TIME
    return torch.stack(seq), target, mask, pats, probe


def completion_error(y, target, mask):
    # fraction of the missing bits that come out with the wrong sign
    return ((torch.sign(y) != target) & mask).sum().item() / mask.sum().item()


def train(plastic, iters, batch=32, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    net = RNN(plastic=plastic)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    for _ in range(iters):
        inputs, target, _, _, _ = make_batch(batch)
        loss = (net(inputs) - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return net, losses


@torch.no_grad()
def evaluate(net, n_episodes=1000, seed=123):
    torch.manual_seed(seed)
    inputs, target, mask, _, _ = make_batch(n_episodes)
    y = net(inputs)
    return (y - target).pow(2).mean().item(), completion_error(y, target, mask)


def smooth(x, k=25):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--iters", type=int, default=1000)
    args = p.parse_args()
    iters = 200 if args.quick else args.iters
    torch.set_num_threads(1)  # tiny ops: thread overhead makes multi-threaded runs several times slower

    nets, curves = {}, {}
    for name, plastic in [("plastic", True), ("non-plastic", False)]:
        nets[name], curves[name] = train(plastic, iters)
        mse, err = evaluate(nets[name])
        print(f"{name:12s} test MSE {mse:.3f}  wrong bits in missing half {err:.3f}")
    print(f"learned eta {nets['plastic'].eta.item():.3f}")

    torch.manual_seed(7)
    inputs, target, mask, pats, probe = make_batch(1)
    with torch.no_grad():
        outs = {k: net(inputs) for k, net in nets.items()}

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1)
    for name, c in curves.items():
        ax.plot(smooth(c), label=name)
    ax.set_xlabel("episode batch")
    ax.set_ylabel("MSE (smoothed)")
    ax.set_title("training loss")
    ax.legend()

    rows = [pats[0, i] for i in range(N_PATTERNS)] + [probe[0], outs["plastic"][0], outs["non-plastic"][0]]
    labels = [f"pattern {i + 1}" for i in range(N_PATTERNS)] + ["probe", "plastic out", "non-plastic out"]
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(torch.stack(rows).numpy(), cmap="bwr", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xticks([])
    ax.set_title("one test episode")
    fig.tight_layout()
    fig.savefig("demo.png", dpi=120)


if __name__ == "__main__":
    main()
