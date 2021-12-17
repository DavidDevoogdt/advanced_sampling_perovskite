# directory is changed to /data, relative paths should be infered from there

root_path = "../.."
cp2k_shell_command = "mpirun cp2k_shell.popt"
debug = True

# cp2k files
CP2K_Path = "calculator/CP2K"
atoms_files = [
    "Pos.xyz",  # phase1
    "Pos2.xyz",  # phase 2
]
cp2k_inp = "orig_cp2k.inp"

# hpc setting
walltime = "72:00:00"
nodes = "nodes=1:ppn=18"

# which modules to test
do_bg = True
