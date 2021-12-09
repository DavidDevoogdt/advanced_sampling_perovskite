from pathlib import Path
import weakref
import time
import sys
import os
#import molmod

import ase
from ase.io import read, write
from ase.io.extxyz import write_extxyz
from ase.calculators.cp2k import CP2K
from ase.calculators.plumed import Plumed
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md import npt
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.parallel import world
from ase.utils import IOContext
from ase import units

import numpy as np

from ase.calculators.calculator import Calculator, all_changes

import functools
print = functools.partial(print, flush=True)


class MyMDLogger(MDLogger):
    """Adds an output column that reports the elapsed time per step"""

    def __init__(self, dyn, atoms, logfile, header=True, stress=False,
                 peratom=False, mode="a"):
        if hasattr(dyn, "get_time"):
            self.dyn = weakref.proxy(dyn)
        else:
            self.dyn = None
        self.atoms = atoms
        global_natoms = atoms.get_global_number_of_atoms()
        self.logfile = self.openfile(logfile, comm=world, mode='a')
        self.stress = stress
        self.peratom = peratom
        if self.dyn is not None:
            self.hdr = "%-9s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        else:
            self.hdr = ""
            self.fmt = ""
        if self.peratom:
            self.hdr += "%12s %12s %12s  %6s" % ("Etot/N[eV]", "Epot/N[eV]",
                                                 "Ekin/N[eV]", "T[K]")
            self.fmt += "%12.4f %12.4f %12.4f  %6.1f"
        else:
            self.hdr += "%12s %12s %12s  %6s" % ("Etot[eV]", "Epot[eV]",
                                                 "Ekin[eV]", "T[K]")
            # Choose a sensible number of decimals
            if global_natoms <= 100:
                digits = 4
            elif global_natoms <= 1000:
                digits = 3
            elif global_natoms <= 10000:
                digits = 2
            else:
                digits = 1
            self.fmt += 3 * ("%%12.%df " % (digits,)) + " %6.1f"
        if self.stress:
            self.hdr += ('      ---------------------- stress [GPa] '
                         '-----------------------')
            self.fmt += 6 * " %10.3f"

        # add timer to log
        self.hdr += '   Time per step [s]'
        self.fmt += '%12.2f'
        self.previous_time = None
        self.fmt += "\n"
        if header:
            self.logfile.write(self.hdr + "\n")

    def __call__(self):
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = self.atoms.get_temperature()
        global_natoms = self.atoms.get_global_number_of_atoms()
        if self.peratom:
            epot /= global_natoms
            ekin /= global_natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000 * ase.units.fs)
            dat = (t,)
        else:
            dat = ()
        dat += (epot + ekin, epot, ekin, temp)
        if self.stress:
            dat += tuple(self.atoms.get_stress(
                include_ideal_gas=True) / ase.units.GPa)

        if self.previous_time is None:
            self.previous_time = time.time()
            time_per_step = 0.0
        else:
            time_per_step = time.time() - self.previous_time
            self.previous_time = time.time()
        dat += (time_per_step,)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()


plumed_input = []

with open("plumed.dat", "r") as f:
    for l in f:
        split_string = l.split("#", 1)
        plumed_input.append(split_string[0].strip())

print(plumed_input)


name = sys.argv[1]


#path_source = Path('/data/gent/vo/000/gvo00003/vsc42365/Libraries')
path_source = Path('/data/gent/vo/000/gvo00003/{}'.format(name))
path_potentials = path_source / 'GTH_POTENTIALS'
path_basis = path_source / 'BASIS_SETS'
path_dispersion = path_source / 'dftd3.dat'

with open("orig_cp2k.inp", "r") as f:
    additional_input = f.read().format(path_basis, path_potentials, path_dispersion)


# initialize atoms object
temperature = 300
if Path('md.traj').exists():  # load previous trajectory
    print('RESTARTING FROM EXISTING TRAJECTORY FILE')

    trajectory = Trajectory('md.traj', 'r')
    atoms = list(trajectory)[-1]  # contains initial velocities
else:
    print('NO EXISTING TRAJECTORY FOUND')
    atoms = read(Path.cwd() / 'Pos.xyz')
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

calculator = CP2K(
    atoms=atoms,
    auto_write=True,
    basis_set=None,
    command='mpirun cp2k_shell.popt',
    cutoff=400 * ase.units.Rydberg,
    stress_tensor=True,
    print_level='LOW',
    inp=additional_input,
    pseudo_potential=None,
    max_scf=None,           # disable
    xc=None,                # disable
    basis_set_file=None,    # disable
    charge=None,            # disable
    potential_file=None,    # disable
    debug=False
)

atoms.calc = calculator
calculator.atoms = atoms
pars = calculator._generate_input()
with open('generated.inp', 'w') as f:
    f.write(pars)

calculator_plumed = Plumed(
    calculator,
    plumed_input,
    2 * ase.units.fs,
    atoms=atoms,
    kT=0.0258,
    log="outplumed",
    restart=False,
    debug=False
)
atoms.calc = calculator_plumed


ptime = 75 * units.fs
bulk = 14e9 * units.Pascal
pfactor = ptime ** 2 * bulk


integrator = npt.NPT(
    atoms,
    timestep=2 * units.fs,
    externalstress=1e5 * units.Pascal,
    ttime=25 * units.fs,
    pfactor=pfactor,
    temperature_K=temperature,
    mask=None,
    trajectory=None,
    logfile=None,
    loginterval=1,
    append_trajectory=False
)

# initialize Trajectory file; only performed by process 0
trajectory = Trajectory(
    'md.traj',
    mode='a',
    atoms=atoms,
    properties=['energy', 'forces', 'stress'],
)
integrator.attach(trajectory.write, step=1)

# initialize Logger
logger = MyMDLogger(integrator, atoms, logfile=sys.stdout)
integrator.attach(logger)

calculator_plumed.register_callbackfn(integrator)

integrator.run(int(72*3600/9))

trajectory = Trajectory('md.traj', 'r')
with open('md.xyz', 'w') as f:
    write_extxyz(f, list(trajectory))
