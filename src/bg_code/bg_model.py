#!/usr/bin/env python

import torch
from src.code.perovskite_energy import PerovskiteEnergy
from bgflow.nn import (DenseNet, SequentialFlow, CouplingFlow, AffineFlow,
                       SplitFlow, InverseFlow, SwapFlow, AffineTransformer)
from bgflow import NormalDistribution
from bgflow import BoltzmannGenerator

from os.path import exists


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
    n_realnvp_blocks = 5
    layers = []
    for _ in range(n_realnvp_blocks):
        layers.append(RealNVP(6, priorDim, hidden=[60,60]))
    flow = SequentialFlow(layers).to(ctx)

    bg = BoltzmannGenerator(prior, flow, target)

    if exists("flow_state_dict.pt"):
        bg.flow.load_state_dict(torch.load("flow_state_dict.pt"))
        bg.flow.eval()
        print("loaded flow params")

    return bg

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
