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










