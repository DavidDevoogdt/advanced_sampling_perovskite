#!/usr/bin/env python

from bgflow.nn.flow.base import Flow
import torch
from src.code.perovskite_energy import PerovskiteEnergy
from bgflow.nn import (DenseNet, SequentialFlow, CouplingFlow, AffineFlow,
                       SplitFlow, InverseFlow, SwapFlow, AffineTransformer,
                       WrapFlow)
from bgflow import NormalDistribution
from bgflow import BoltzmannGenerator

from os.path import exists
import src

import numpy as np


def bg_model():
    # if src.config.debug:
    print("## creating CP2K", flush=True)
    target = PerovskiteEnergy()
    #target.add_init_configuration("Pos2.xyz")  # second phase

    ctx = target.ctx

    # if src.config.debug:
    print("## creating prior+flow", flush=True)
    # throw away 3 translation dims and 3 rotations
    priorDim = target.init_state.shape[1]
    mean = torch.zeros(priorDim).to(ctx)  # cuda version
    prior = NormalDistribution(priorDim, mean=mean)

    # having a flow and a prior, we can now define a Boltzmann Generator
    n_realnvp_blocks = src.config.bg_rNVP_layers
    layers = []

    types = np.array([0, 1, 2, 3, 4, 4, 4])
    sizes = np.array([3, 3, 3, 3, 3])

    split_into_n = SplitFlow(*sizes[types])

    layers.append(split_into_n)
    for _ in range(n_realnvp_blocks):
        layers.append(
            RealNVP(hidden=[
                src.config.bg_NN_layers for _ in range(src.config.bg_NN_nodes)
            ],
                    types=types,
                    sizes=sizes))
    layers.append(InverseFlow(split_into_n))

    mu = target.init_state[0].clone().detach() * 0
    mu[3:6] = 90  #angle

    sigma = target.init_state[0].clone().detach()
    sigma = sigma * 0 + 1
    sigma[3:6] = sigma[3:6] * 10  #variance of 10 degrees for angle

    layers.append(InverseFlow(normData(mu, sigma)))

    flow = SequentialFlow(layers).to(ctx)

    bg = BoltzmannGenerator(prior, flow, target)

    if exists("flow_state_dict.pt"):
        bg.flow.load_state_dict(torch.load("flow_state_dict.pt"))
        bg.flow.eval()
        print("loaded flow params")

    return bg


class normData(Flow):
    def __init__(self, mu=None, sigma=None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def _forward(self, xs, **kwargs):
        if self.mu is not None:
            xs = xs - self.mu
        if self.sigma is not None:
            xs = xs / self.sigma

        dlogp = 0

        return (xs, dlogp)

    def _inverse(self, xs, **kwargs):
        if self.sigma is not None:
            xs = xs * self.sigma
        if self.mu is not None:
            xs = xs + self.mu

        dlogp = 0

        return (*xs, dlogp)


class RealNVP(SequentialFlow):
    def __init__(self, hidden, types, sizes):

        #cell length xyz, cell angles, Cs, Pb, I,I,I
        self.types = types
        self.sizes = sizes

        self.ntypes = np.max(self.types) + 1

        class tinfo:
            pass

        self.typeinfo = []
        for i in range(self.ntypes):

            t = tinfo()
            t.i = i
            t.members = np.argwhere(self.types == i).flatten().tolist()
            t.other_members = np.argwhere(self.types != i).flatten().tolist()

            t.size = self.sizes[i]

            self.typeinfo.append(t)

        self.hidden = hidden
        super().__init__(self._create_layers())

    def _create_layers(self):

        layers = []

        sum_d = []
        mul_d = []

        #create densenets shift and scale from every type to every other type
        for t in self.typeinfo:
            sub_sum = []
            sub_mul = []

            for ot in self.typeinfo:
                sub_sum.append(
                    DenseNet([t.size, *self.hidden, ot.size],
                             activation=torch.nn.ReLU()))

                sub_mul.append(
                    DenseNet([t.size, *self.hidden, ot.size],
                             activation=torch.nn.ReLU()))

            sum_d.append(sub_sum)
            mul_d.append(sub_mul)

        for t in self.typeinfo:
            #apply map to different species
            for m in t.members:

                for om in t.other_members:

                    t2 = self.typeinfo[self.types[om]]

                    l = CouplingFlow(AffineTransformer(
                        shift_transformation=sum_d[t.i][t2.i],
                        scale_transformation=mul_d[t.i][t2.i]),
                                     transformed_indices=(m, ),
                                     cond_indices=(om, ))

                    layers.append(l)

            #apply map to equal species, sum first
            for m in t.members:
                for om in t.members:
                    if m != om:
                        l = CouplingFlow(AffineTransformer(
                            shift_transformation=sum_d[t.i][t.i],
                            scale_transformation=None),
                                         transformed_indices=(m, ),
                                         cond_indices=(om, ))

                        layers.append(l)

            #apply map to equal species, multiply
            for m in t.members:
                for om in t.members:
                    if m != om:
                        l = CouplingFlow(AffineTransformer(
                            shift_transformation=None,
                            scale_transformation=mul_d[t.i][t.i]),
                                         transformed_indices=(m, ),
                                         cond_indices=(om, ))

                        layers.append(l)

        return layers