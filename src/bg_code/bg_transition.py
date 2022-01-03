from bgflow.bg import BoltzmannGenerator
from bgflow.distribution.normal import NormalDistribution
from bgflow.nn.flow.base import Flow
from src.bg_code.bg_model import RealNVP
from src.code.perovskite_energy import PerovskiteEnergy
import numpy as np
import torch
import ase

from ase.io.extxyz import write_extxyz
from ase.io.proteindatabank import write_proteindatabank


def bg_transition(bg: BoltzmannGenerator):
    # choose 2 init points, interpolate in prior space, tranform back to real space, save atoms as xyz
    flow: RealNVP = bg.flow
    target: PerovskiteEnergy = bg._target
    init = target.init_state.clone().detach()
    (init_prior, _) = flow.forward(init, inverse=True)

    a_prior = init_prior[0, :]
    b_prior = init_prior[1, :]

    atom_list = []

    for w in np.linspace(0, 1, 200):
        c_prior = torch.lerp(a_prior, b_prior, w)
        (c, _) = flow.forward(c_prior)
        atom = target.tensor_to_atoms(c).copy()
        atom_periodic = atom.repeat(3)
        atom_list.append(atom_periodic)

    with open('interp.pdb', 'w') as f:
        write_proteindatabank(f, atom_list)
