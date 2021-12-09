from .ASEbridge import ASEbridge

from bgflow import Energy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


__all__ = ["PerovskiteEnergy"]

# test implememtation, not connectep at the moment


class PerovskiteEnergy(Energy):
    # name = vsc_account number
    def __init__(self, dim, name, temp):
        super().__init__(dim)

        self.ab = ASEbridge(name)
        self.name = name
        self.temp = temp
        self.calc = self.ab.get_CP2K_calculator()

    def _energy(self, x):

        at = self.calc.atoms

        for i in x[:, 0:]:
            at.set_positions(i)
            MaxwellBoltzmannDistribution(at, temperature_K=self.temp)

        return e1 + e2
