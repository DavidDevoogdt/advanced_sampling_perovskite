#!/usr/bin/env python

import torch
from src.code.perovskite_energy import PerovskiteEnergy
from bgflow.nn import (DenseNet, SequentialFlow, CouplingFlow, AffineFlow,
                       SplitFlow, InverseFlow, SwapFlow, AffineTransformer)
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
    for _ in range(n_realnvp_blocks):
        layers.append(
            RealNVP(hidden=[
                src.config.bg_NN_layers for _ in range(src.config.bg_NN_nodes)
            ]))
    flow = SequentialFlow(layers).to(ctx)

    bg = BoltzmannGenerator(prior, flow, target)

    if exists("flow_state_dict.pt"):
        bg.flow.load_state_dict(torch.load("flow_state_dict.pt"))
        bg.flow.eval()
        print("loaded flow params")

    return bg


class RealNVP(SequentialFlow):
    def __init__(self, hidden):

        #work out all the details of the layers

        #cell length xyz, cell angles, Cs, Pb, I,I,I
        self.types = np.array([0, 1, 2, 3, 4, 4, 4])
        self.sizes = np.array([3, 3, 3, 3, 3])

        ntypes = np.max(self.types) + 1
        nmemb = len(self.types)
        self.all = np.cumsum(np.ones(nmemb, dtype=int)) - 1
        self.totaldim = np.sum(self.sizes[self.types])

        class tinfo:
            pass

        self.typeinfo = []
        for i in range(ntypes):

            t = tinfo()
            t.i = i
            t.members = np.where(self.types == i)[0]
            t.size = self.sizes[i]
            t.nsize = self.totaldim - self.sizes[i]

            self.typeinfo.append(t)

        self.hidden = hidden
        super().__init__(self._create_layers())

    def _create_layers(self):

        split_into_n = SplitFlow(*self.sizes[self.types])
        layers = [split_into_n]

        sum_d = []
        mul_d = []

        #create densenets shift and scale
        for t in self.typeinfo:
            sum_d.append(
                DenseNet([t.size, *self.hidden, t.nsize],
                         activation=torch.nn.ReLU()))
            mul_d.append(
                DenseNet([t.size, *self.hidden, t.nsize],
                         activation=torch.nn.ReLU()))

        #make the flows and apply per type
        for op in ['sum', 'mul']:

            for t in self.typeinfo:
                #apply all the shift flows
                for m in t.members:
                    ti = self.all.tolist().copy()
                    ti.remove(m)

                    shiftt = None
                    scalet = None
                    if op == "sum":
                        # shiftt = DenseNet([t.size, *self.hidden, t.nsize],
                        #                   activation=torch.nn.ReLU())
                        shiftt = sum_d[t.i]
                    if op == "mul":
                        # scalet = DenseNet([t.size, *self.hidden, t.nsize],
                        #                   activation=torch.nn.ReLU())
                        scalet = mul_d[t.i]

                    l = CouplingFlow(AffineTransformer(
                        shift_transformation=shiftt,
                        scale_transformation=scalet),
                                     transformed_indices=tuple(ti),
                                     cond_indices=(m, ))

                    layers.append(l)

        layers.append(InverseFlow(split_into_n))

        return layers