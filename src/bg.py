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
from bgflow.nn import (DenseNet, SequentialFlow, CouplingFlow, AffineFlow,
                       SplitFlow, InverseFlow, SwapFlow, AffineTransformer)
from bgflow import NormalDistribution
from bgflow import BoltzmannGenerator
from bgflow.utils.types import is_list_or_tuple
from bgflow.utils.train import IndexBatchIterator
import bgflow

from os.path import exists


def bg(temp=300):
    target = PerovskiteEnergy(temp)
    target.add_init_configuration("Pos2.xyz")  # second phase

    ctx = target.ctx

    target_sampler = GaussianMCMCSampler(target, init_state=target.init_state)

    # time intensive step, load from disk
    if exists('data.pt'):
        data = torch.load('data.pt')
    else:
        data = target_sampler.sample(5)
        torch.save(data, 'data.pt')

    # throw away 3 translation dims and 3 rotations
    priorDim = target.init_state.shape[1]
    mean = torch.zeros(priorDim).to(ctx)  # cuda version
    prior = NormalDistribution(priorDim, mean=mean)

    # having a flow and a prior, we can now define a Boltzmann Generator
    n_realnvp_blocks = 5
    layers = []
    for _ in range(n_realnvp_blocks):
        layers.append(RealNVP(6, priorDim, hidden=[30, 30]))
    flow = SequentialFlow(layers).to(ctx)

    #
    bg = BoltzmannGenerator(prior, flow, target)

    # first training
    nll_optimizer = torch.optim.Adam(bg.parameters(), lr=1e-3)
    nll_trainer = bgflow.KLTrainer(bg, optim=nll_optimizer, train_energy=False)

    nll_trainer.train(n_iter=100,
                      data=data,
                      batchsize=3,
                      n_print=2,
                      w_energy=0.0)

    # mixed training
    mixed_optimizer = torch.optim.Adam(bg.parameters(), lr=1e-4)
    mixed_trainer = bgflow.KLTrainer(bg,
                                     optim=mixed_optimizer,
                                     train_energy=True)

    mixed_trainer.train(n_iter=2000,
                        data=data,
                        batchsize=1000,
                        n_print=100,
                        w_energy=0.1,
                        w_likelihood=0.9,
                        clip_forces=20.0)


class RealNVP(SequentialFlow):
    def __init__(self, dim1, totaldim, hidden):
        self.dim1 = dim1
        self.tdim = totaldim
        self.hidden = hidden
        super().__init__(self._create_layers())

    def _create_layers(self):
        dim_channel1 = self.dim1
        dim_channel2 = self.tdim - self.dim1
        split_into_2 = SplitFlow(dim_channel1, dim_channel2)

        layers = [
            # -- split
            split_into_2,
            # --transform
            self._coupling_block(dim_channel1, dim_channel2),
            SwapFlow(),
            self._coupling_block(dim_channel2, dim_channel1),
            SwapFlow(),
            # -- merge
            InverseFlow(split_into_2)
        ]
        return layers

    def _dense_net(self, dim1, dim2):

        layers = [dim1, *self.hidden, dim2]
        return DenseNet(layers, activation=torch.nn.ReLU())

    def _coupling_block(self, dim1, dim2):
        return CouplingFlow(
            AffineTransformer(shift_transformation=self._dense_net(dim1, dim2),
                              scale_transformation=self._dense_net(dim1,
                                                                   dim2)))
