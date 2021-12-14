#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.code.perovskite_energy import PerovskiteEnergy
from src.code.ASEbridge import ASEbridge
from bgflow.utils.types import assert_numpy
from bgflow import GaussianMCMCSampler
from bgflow import Energy, Sampler
from torch.distributions.chi2 import Chi2
from torch.distributions.gamma import Gamma
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
from bgflow import NormalDistribution
from bgflow import BoltzmannGenerator
from bgflow.utils.types import is_list_or_tuple
from bgflow.utils.train import IndexBatchIterator
import bgflow


def bg():

    name = "vsc43693"  # sys.argv[1]
    temp = 300

    print(name)

    target = PerovskiteEnergy(name, temp)

    ctx = target.ctx
    init_state = target.init_state

    target_sampler = GaussianMCMCSampler(target, init_state=init_state)
    data = target_sampler.sample(50)

    priorDim = 2
    prior = NormalDistribution(priorDim)

    # define a flow with RNVP coupling layers

    class RealNVP(bgflow.SequentialFlow):

        def __init__(self, dim, hidden):
            self.dim = dim
            self.hidden = hidden
            super().__init__(self._create_layers())

        def _create_layers(self):
            dim_channel1 = self.dim//2
            dim_channel2 = self.dim - dim_channel1
            split_into_2 = bgflow.SplitFlow(dim_channel1, dim_channel2)

            layers = [
                # -- split
                split_into_2,
                # --transform
                self._coupling_block(dim_channel1, dim_channel2),
                bgflow.SwapFlow(),
                self._coupling_block(dim_channel2, dim_channel1),
                # -- merge
                bgflow.InverseFlow(split_into_2)
            ]
            return layers

        def _dense_net(self, dim1, dim2):
            return bgflow.DenseNet(
                [dim1, *self.hidden, dim2],
                activation=torch.nn.ReLU()
            )

        def _coupling_block(self, dim1, dim2):
            return bgflow.CouplingFlow(bgflow.AffineTransformer(
                shift_transformation=self._dense_net(dim1, dim2),
                scale_transformation=self._dense_net(dim1, dim2)
            ))

    # having a flow and a prior, we can now define a Boltzmann Generator
    n_realnvp_blocks = 5
    layers = []

    dim_cell_lengths = 3
    dim_cell_angle = 3
    dim_cell_cartesion = 15
    totaldim = dim_cell_lengths+dim_cell_angle+dim_cell_cartesion

    split_flow = bgflow.SplitFlow(
        dim_cell_lengths, dim_cell_angle, dim_cell_cartesion)

    for i in range(n_realnvp_blocks):
        layers.append(RealNVP(totaldim, hidden=[128, 128, 128]))
    layers.append(split_flow)

    flow = bgflow.SequentialFlow(layers).to(ctx)

    #

    bg = BoltzmannGenerator(prior, flow, target)

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

    # plot_bg(bg, target, dim=dim)
    # plot_weighted_energy_estimate(bg, target, dim=dim)

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
