from os import error
from ase.units import C
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch._C import InferredType
from .ASEbridge import ASEbridge
from bgflow import Energy

import torch
import numpy as np
import math
import src
import sys

__all__ = ["PerovskiteEnergy"]


# test implememtation, not connectep at the moment
class PerovskiteEnergy(Energy):
    # name = vsc_account number
    def __init__(self):
        self.ab = ASEbridge()
        self.calc = self.ab.get_CP2K_calculator()

        # context tensor torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        self.ctx = torch.zeros([], device=device, dtype=dtype)

        self.init_state = None
        self.add_init_configuration()

        self.n = 0

        self.max_ener = 0.0

        super().__init__(self.totaldims)

    def add_init_configuration(self, atom_names=None):
        #at = read("{}{}".format(CP2K_Path, name))
        atom_list = self.ab.get_atoms(atom_names)
        for at in atom_list:
            print("added atom {}".format(at))
            cx = self.atom_to_tensor(at)

            if self.init_state is None:
                self.totaldims = cx.shape[0]
                self.init_state = cx.reshape(1, self.totaldims)
                self.init_atom = at
                self.dims = at.positions.shape
            else:
                self.init_state = torch.cat(
                    [self.init_state,
                     cx.reshape(1, self.totaldims)], dim=0)

    def tensor_to_atoms(self, cx):
        at = self.init_atom
        c = cx[:6].clone().cpu().detach().numpy()
        c[3:6] = (c[3:6] + 1) * 90

        x = cx[6:].clone().cpu().detach().numpy()

        at.cell = at.cell.fromcellpar(c)
        at.set_positions(x.reshape(self.dims))

        at.calc = self.calc
        return at

    def atom_to_tensor(self, at):
        x = at.get_positions()
        c = at.get_cell().cellpar()
        c[3:6] = c[3:6] / 90 - 1  #mean 0

        cx = torch.Tensor(np.concatenate((c, x.flatten()))).to(self.ctx)
        return cx

    def _energy(self, x):
        # calculates CP2K energy for all configurations in x

        if x.dim() == 1:
            x = x.reshape((1, self.totaldims))

        n = x.shape[0]

        ener = x[:, [0]].clone() * 0

        print("{:4d}|".format(self.n), end='', flush=True)

        # this can be parallised
        for i in range(n):
            try:
                at = self.tensor_to_atoms(x[i, :])

                try:  #prone to error
                    ener[i] = at.get_potential_energy()
                except Exception as e1:  #energy calculation faild due to bad config

                    if src.config.debug == True:
                        print(
                            "something went wrong with CP2K calculator, resetting\n{}"
                            .format(e1),
                            flush=True)
                    try:
                        del self.calc  #calculator is corrupted, deconstruct if possible
                    except:
                        pass

                    #setup a new calculator,assume the error is due to a bad configuration
                    self.calc = self.ab.get_CP2K_calculator()
                    ener[i] = self.max_ener

            except:  #bad atoms cought
                ener[i] = self.max_ener

            print("{}:{:10.4f} ".format(i, float(ener[i])),
                  end='',
                  flush=True)

        print("", flush=True)
        self.n = self.n + 1

        return ener
