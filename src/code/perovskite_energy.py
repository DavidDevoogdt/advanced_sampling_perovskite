from os import error
from ase.units import C
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch._C import InferredType
from .ASEbridge import ASEbridge
from bgflow import Energy
from config import CP2K_Path
import torch
import numpy as np
import math
import config

__all__ = ["PerovskiteEnergy"]


# test implememtation, not connectep at the moment
class PerovskiteEnergy(Energy):
    # name = vsc_account number
    def __init__(self, temp):
        self.ab = ASEbridge()
        self.temp = temp
        self.calc = self.ab.get_CP2K_calculator()

        # context tensor torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        self.ctx = torch.zeros([], device=device, dtype=dtype)

        self.init_state = None
        self.add_init_configuration()

        super().__init__(self.totaldims)

    def add_init_configuration(self, atom_names=None):
        #at = read("{}{}".format(CP2K_Path, name))
        atom_list = self.ab.get_atoms(atom_names)
        for at in atom_list:
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
        c = cx[:6].detach().cpu().numpy()

        c[3:6] = (c[3:6] + 1) * 90

        x = cx[6:].detach().cpu().numpy()

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

    def _energy(self, x, temperature=None):
        # calculates CP2K energy for all configurations in x
        if temperature is None:
            temperature = self.temp

        if x.dim() == 1:
            x = x.reshape((1, self.totaldims))

        n = x.shape[0]

        ener = x[:, [0]].clone() * 0

        # this can be parallised
        for i in range(n):

            try:  #prone to error
                at = self.tensor_to_atoms(x[i, :])
                MaxwellBoltzmannDistribution(at, temperature_K=temperature)
                ener[i] = at.get_potential_energy()
            except Exception as e1:  #energy calculation faild due to bad config

                if config.debug == True:
                    print(
                        "something went wrong with CP2K calculator, resetting\n{}"
                        .format(e1),
                        flush=True)
                try:
                    del self.calc
                except Exception as e2:
                    pass

                #setup a new calculator,assume the error is due to a bad configuration
                self.calc = self.ab.get_CP2K_calculator()
                ener[i] = math.inf
        return ener
