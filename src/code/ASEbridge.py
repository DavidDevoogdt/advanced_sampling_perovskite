from ase.io import read
from pathlib import Path

import ase
from ase.calculators.cp2k import CP2K
from ase import units

from config import cp2k_shell_command
from config import debug
from config import CP2K_Path


class ASEbridge:

    def __init__(self, name) -> None:
        self.name = name

    def get_CP2K_calculator(self):

        path_source = Path(
            "{}/Libraries/".format(CP2K_Path))
        path_potentials = path_source / 'GTH_POTENTIALS'
        path_basis = path_source / 'BASIS_SETS'
        path_dispersion = path_source / 'dftd3.dat'

        CP2K_path = "./calculator/CP2K"

        with open("{}/orig_cp2k.inp".format(CP2K_path), "r") as f:
            additional_input = f.read().format(path_basis, path_potentials, path_dispersion)

        #temperature = 300
        atoms = read("{}/Pos.xyz".format(CP2K_path))
        #MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

        calculator = CP2K(
            atoms=atoms,
            auto_write=True,
            basis_set=None,
            command=cp2k_shell_command,
            cutoff=400 * units.Rydberg,
            stress_tensor=True,
            print_level='LOW',
            inp=additional_input,
            pseudo_potential=None,
            max_scf=None,           # disable
            xc=None,                # disable
            basis_set_file=None,    # disable
            charge=None,            # disable
            potential_file=None,    # disable
            debug=debug
        )

        atoms.calc = calculator
        calculator.atoms = atoms
        # pars = calculator._generate_input()
        # with open('generated.inp', 'w') as f:
        #     f.write(pars)

        return calculator

    def get_Plumed_CP2K_calculator(self):
        raise NotImplementedError
