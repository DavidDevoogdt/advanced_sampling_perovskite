from ase.io import read
from pathlib import Path

import ase
from ase.calculators.cp2k import CP2K
from ase import units

import src


class ASEbridge:
    def __init__(self) -> None:
        pass

    def get_CP2K_calculator(self):

        path_source = Path("{}/{}/Libraries/".format(src.config.root_path,
                                                     src.config.cp2k_path))
        path_potentials = path_source / 'GTH_POTENTIALS'
        path_basis = path_source / 'BASIS_SETS'
        path_dispersion = path_source / 'dftd3.dat'

        with open(
                "{}/{}/{}".format(src.config.root_path, src.config.cp2k_path,
                                  src.config.cp2k_inp), "r") as f:
            additional_input = f.read().format(path_basis, path_potentials,
                                               path_dispersion)

        # atoms = read("{}/Pos.xyz".format(CP2K_path))

        calculator = CP2K(
            atoms=None,
            auto_write=True,
            basis_set=None,
            command=src.config.cp2k_shell,
            cutoff=400 * units.Rydberg,
            stress_tensor=True,
            print_level='LOW',
            inp=additional_input,
            pseudo_potential=None,
            max_scf=None,  # disable
            xc=None,  # disable
            basis_set_file=None,  # disable
            charge=None,  # disable
            potential_file=None,  # disable
            debug=src.config.debug)

        return calculator

    def get_atoms(self, atom_files=None):
        if atom_files is None:
            atom_files = src.config.atoms_files
        if isinstance(atom_files, str):
            atom_files = [
                atom_files,
            ]
        return [
            read("{}/{}/{}".format(src.config.root_path, src.config.cp2k_path,
                                   name)) for name in atom_files
        ]

    def get_Plumed_CP2K_calculator(self):
        raise NotImplementedError
