#!/usr/bin/env python

from bgflow.utils.train import IndexBatchIterator
from bgflow.utils.types import is_list_or_tuple
from bgflow import BoltzmannGenerator
from bgflow import NormalDistribution
from bgflow.nn import (
    DenseNet,
    SequentialFlow,
    CouplingFlow,
    AffineFlow,
    SplitFlow,
    InverseFlow,
    SwapFlow,
    AffineTransformer
)
from torch.distributions.gamma import Gamma
from torch.distributions.chi2 import Chi2
from bgflow import Energy, Sampler
from bgflow import GaussianMCMCSampler
from bgflow.utils.types import assert_numpy


from src.code.ASEbridge import ASEbridge
from src.code.perovskite_energy import PerovskiteEnergy

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys


name = "vsc43693"  # sys.argv[1]
temp = 300

print(name)

target = PerovskiteEnergy(name, temp)


# define a MCMC sampler to sample from the target energy


init_state = target.atom_to_tensor()
# init_state = torch.cat([init_state, torch.Tensor(
#     init_state.shape[0], dim-2).normal_()], dim=-1)
target_sampler = GaussianMCMCSampler(target, init_state=init_state)

# sample some data

data = target_sampler.sample(50000)

# plot_samples(data)


class HypersphericalPrior(Energy, Sampler):

    def __init__(self, dim, concentration=1.):
        super().__init__(dim)
        r = np.sqrt(dim) / 2
        rate = concentration / r
        self._gamma = Gamma(concentration, rate)

    def _energy(self, x):
        d2 = x.pow(2).sum(dim=-1, keepdim=True)
        d = (d2 + 1e-7).sqrt()
        return -self._gamma.log_prob(d)

    def _sample(self, n_samples):
        x = torch.Tensor(n_samples, self._dim).normal_()
        d2 = x.pow(2).sum(dim=-1, keepdim=True)
        d = (d2 + 1e-7).sqrt()
        r = x / d
        s = self._gamma.sample((n_samples, 1))
#         print(s)
        return r * s

# now set up a prior


print("expected crash :p")

prior = NormalDistribution(dim)
# prior = HypersphericalPrior(dim, concentration=10)
# define a flow with RNVP coupling layers


# here we aggregate all layers of the flow
layers = []

# start with a splitting layer which splits the input tensor into two
# flow channels with tensors of half dimensionality
layers.append(SplitFlow(dim // 2))


# now add coupling layers
n_coupling_layers = 4
for _ in range(n_coupling_layers):

    # we need to swap dimensions for the mixing
    layers.append(SwapFlow())

    # now set up a coupling block
    layers.append(CouplingFlow(
        # we use a affine transformation to transform the RHS conditioned on the LHS
        AffineTransformer(
            # use simple dense nets for the affine shift/scale
            shift_transformation=DenseNet(
                [dim // 2, 64, 64, dim // 2], activation=torch.nn.ReLU()),
            scale_transformation=DenseNet(
                [dim // 2, 64, 64, dim // 2], activation=torch.nn.ReLU())
        )
    ))

# finally, we have to merge the two channels again into one tensor
layers.append(InverseFlow(SplitFlow(dim // 2)))

# now define the flow as a sequence of all operations stored in layers
flow = SequentialFlow(layers)

# having a flow and a prior, we can now define a Boltzmann Generator


bg = BoltzmannGenerator(prior, flow, target)

# initial bg should not be useful
plot_bg(bg, target, dim=dim)

plot_weighted_energy_estimate(bg, target)

# first training


class LossReporter:
    """
        Simple reporter use for reporting losses and plotting them.
    """

    def __init__(self, *labels):
        self._labels = labels
        self._n_reported = len(labels)
        self._raw = [[] for _ in range(self._n_reported)]

    def report(self, *losses):
        assert len(losses) == self._n_reported
        for i in range(self._n_reported):
            self._raw[i].append(assert_numpy(losses[i]))

    def plot(self, n_smooth=10):
        fig, axes = plt.subplots(self._n_reported, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        fig.set_size_inches((8, 4 * self._n_reported), forward=True)
        for i, (label, raw, axis) in enumerate(zip(self._labels, self._raw, axes)):
            raw = assert_numpy(raw).reshape(-1)
            kernel = np.ones(shape=(n_smooth,)) / n_smooth
            smoothed = np.convolve(raw, kernel, mode="valid")
            axis.plot(smoothed)
            axis.set_ylabel(label)
            if i == self._n_reported - 1:
                axis.set_xlabel("Iteration")

    def recent(self, n_recent=1):
        return np.array([raw[-n_recent:] for raw in self._raw])


# initial training with likelihood maximization on data set
n_batch = 32
batch_iter = IndexBatchIterator(len(data), n_batch)

optim = torch.optim.Adam(bg.parameters(), lr=5e-3)

n_epochs = 5
n_report_steps = 50

reporter = LossReporter("NLL")

for epoch in range(n_epochs):
    for it, idxs in enumerate(batch_iter):
        batch = data[idxs]

        optim.zero_grad()

        # negative log-likelihood of the batch is equal to the energy of the BG
        nll = bg.energy(batch).mean()
        nll.backward()

        reporter.report(nll)

        optim.step()

        if it % n_report_steps == 0:
            print("\repoch: {0}, iter: {1}/{2}, NLL: {3:.4}".format(
                epoch,
                it,
                len(batch_iter),
                *reporter.recent(1).ravel()
            ), end="")
reporter.plot()

# bg after ML training

plot_bg(bg, target, dim=dim)
plot_weighted_energy_estimate(bg, target, dim=dim)

# train with convex mixture of NLL and KL loss


n_kl_samples = 128
n_batch = 128
batch_iter = IndexBatchIterator(len(data), n_batch)

optim = torch.optim.Adam(bg.parameters(), lr=5e-3)

n_epochs = 5
n_report_steps = 50

# mixing parameter
lambdas = torch.linspace(1., 0.5, n_epochs)

reporter = LossReporter("NLL", "KLL")

torch.linspace(1., 0.5, n_epochs)


for epoch, lamb in enumerate(lambdas):
    for it, idxs in enumerate(batch_iter):
        batch = data[idxs]

        optim.zero_grad()

        # negative log-likelihood of the batch is equal to the energy of the BG
        nll = bg.energy(batch).mean()

        # aggregate weighted gradient
        (lamb * nll).backward()

        # kl divergence to the target
        kll = bg.kldiv(n_kl_samples).mean()

        # aggregate weighted gradient
        ((1. - lamb) * kll).backward()

        reporter.report(nll, kll)

        optim.step()

        if it % n_report_steps == 0:
            print("\repoch: {0}, iter: {1}/{2}, lambda: {3}, NLL: {4:.4}, KLL: {5:.4}".format(
                epoch,
                it,
                len(batch_iter),
                lamb,
                *reporter.recent(1).ravel()
            ), end="")

reporter.plot()

# bg after ML + KL training

plot_bg(bg, target, dim=dim)

plot_weighted_energy_estimate(bg, target, dim=dim)
