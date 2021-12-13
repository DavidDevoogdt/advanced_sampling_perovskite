from .ASEbridge import ASEbridge

from bgflow import Energy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import torch

__all__ = ["PerovskiteEnergy"]

# test implememtation, not connectep at the moment


class PerovskiteEnergy(Energy):
    # name = vsc_account number
    def __init__(self, name, temp):
        self.ab = ASEbridge(name)
        self.name = name
        self.temp = temp
        self.calc = self.ab.get_CP2K_calculator()

        a = self.calc.get_potential_energy()
        print("enegy works {}".format(a))

        self.dims = self.calc.atoms.get_positions().shape

        super().__init__(self.dims)

    def tensor_to_atoms(self, x):
        at = self.calc.atoms.copy()

        at.set_positions(x)
        MaxwellBoltzmannDistribution(at, temperature_K=self.temp)

        at.calc = self.calc
        return at

    def atom_to_tensor(self, at=None):
        if at is None:
            at = self.calc.atoms

        x = at.get_positions()
        x = torch.Tensor(x)
        return x

    def _energy(self, x):

        at = self.tensor_to_atoms(x)
        MaxwellBoltzmannDistribution(at, temperature_K=self.temp)

        ener = at.get_potential_energy()

        print("enegy for {} = {}".format(x, ener))

        return ener
