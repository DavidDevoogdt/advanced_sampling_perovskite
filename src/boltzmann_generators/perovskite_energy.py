import ASEbridge


from bgflow import Energy


__all__ = ["PerovskiteEnergy"]

# test implememtation, not connected at the moment


class PerovskiteEnergy(Energy):
    # name = vsc_account number
    def __init__(self, dim, name, a=0, b=-4.0, c=1.0):
        super().__init__(dim)

        self.ab = ASEbridge()
        self.calc = self.ab.get_calculator(name)

        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d.pow(2) + self._c * d.pow(4)
        e2 = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return e1 + e2
