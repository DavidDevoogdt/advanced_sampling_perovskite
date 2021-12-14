from ase.units import C
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from .ASEbridge import ASEbridge
from bgflow import Energy
from config import CP2K_Path
import torch
import numpy as np


__all__ = ["PerovskiteEnergy"]

# test implememtation, not connectep at the moment


class PerovskiteEnergy(Energy):
    # name = vsc_account number
    def __init__(self, temp):
        self.ab = ASEbridge()
        self.temp = temp
        self.calc = self.ab.get_CP2K_calculator()
        self.dims = self.calc.atoms.get_positions().shape
        self.totaldims = int(np.prod(np.array(self.dims))
                             ) + 6  # also cell parmams

        super().__init__(self.totaldims)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        # context tensor torch
        self.ctx = torch.zeros([], device=device, dtype=dtype)

        self.init_state = self.atom_to_tensor().reshape(
            (1, self.totaldims))  # this a nd array with initial states

    def add_init_configuration(self, name):
        at = read("{}{}".format(CP2K_Path, name))
        cx = self.atom_to_tensor(at)
        self.init_state = torch.cat(
            [self.init_state, cx.reshape(1, self.totaldims)], dim=0)

    def tensor_to_atoms(self, cx):
        at = self.calc.atoms.copy()

        c = cx[:6].detach().cpu().numpy()
        x = cx[6:].detach().cpu().numpy()

        at.cell = at.cell.fromcellpar(c)
        at.set_positions(x.reshape(self.dims))

        at.calc = self.calc
        return at

    def atom_to_tensor(self, at=None):
        # encode the parameters in a torch array
        if at is None:
            at = self.calc.atoms

        x = at.get_positions()
        c = at.get_cell().cellpar()

        cx = torch.Tensor(np.concatenate((c, x.flatten()))).to(self.ctx)
        return cx

    def _energy(self, x, temperature=None):
        # calculates CP2K energy for all configurations in x
        if temperature is None:
            temperature = self.temp

        if x.dim() == 1:
            x = x.reshape((1, self.totaldims))

        n = x.shape[0]

        ener = x[:, [0]].clone()*0

        # this can be parallised
        for i in range(n):
            at = self.tensor_to_atoms(x[i, :])
            MaxwellBoltzmannDistribution(at, temperature_K=temperature)

            ener[i] = at.get_potential_energy()
        return ener
