
from ase.io.trajectory import Trajectory

from ase.io.extxyz import write_extxyz
from ase.io.cif import write_cif
from ase.io.proteindatabank import write_proteindatabank

from ase.visualize import view


trajectory = Trajectory('md.traj', 'r')

print(list(trajectory))

t = "PDB"



# with open('md.{}'.format(t), 'w') as f:
#     if t == "xyz":
#         write_extxyz(f, list(trajectory))
#     elif t == "PDB":
#         write_proteindatabank(f, list(trajectory))
#     elif t == "CIF":
#         write_cif(f, list(trajectory))
