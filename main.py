from genericpath import exists
from loader import main as l

import pathlib
import time
import os
import sys
import argparse
import pickle
import src

# sets up a new folder for all the output. If run on hpc, the python scripts is called with qsub. Config file is copied as reference


def main(args):

    rp = pathlib.Path(__file__).parent.resolve()

    data_folder = rp / 'data' / args.foldername

    if not exists(data_folder):
        data_folder.mkdir(parents=True)
    #os.system("cp config.py data/{}/config.py".format(args.foldername))
    os.chdir(data_folder.resolve())
    pickle.dump(args, open('config.pickle', 'wb'))
    with open('config.txt', 'w') as f:
        f.write(str(args))

    if args.hpc:  # hpc user
        write_submit(rp, args)
        os.system('qsub submitscript.sh')
        print("job submitted, folder name {}".format(args.foldername))
    else:  # regular user
        print("folder name {}, running directly".format(args.foldername))
        src.config = args
        l()


def write_submit(rp, args):
    with open(rp / "calculator" / 'submitscript_template.sh') as f:
        submitscript = f.read().format(args.hpc_walltime, args.hpc_nodes,
                                       str(rp.resolve()))
    with open("submitscript.sh", 'w') as f:
        f.write(submitscript)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulations')

    group0 = parser.add_argument_group('general settings', '')

    group0.add_argument('-d', '--debug', action='store_true', help='')
    group0.add_argument('--do_bg', default=True)

    group1 = parser.add_argument_group('folders and paths', 'default paths')
    group1.add_argument(
        '-n',
        '--foldername',
        type=str,
        default=time.strftime("%Y-%m-%d_%H-%M-%S"),
        help='foldername for all output. Default -> current time')

    #no need to specify these:
    group1.add_argument('--root_path', default="../..", help='')
    group1.add_argument('--cp2k_path', default="calculator/CP2K", help='')
    group1.add_argument('--cp2k_shell',
                        default="mpirun cp2k_shell.popt",
                        help='')
    group1.add_argument(
        '--atoms_files',
        default=[
            "Pos.xyz",  # phase1
            "Pos2.xyz",  # phase 2
        ],
        help='')
    group1.add_argument('--cp2k_inp', default="orig_cp2k.inp", help='')

    group3 = parser.add_argument_group('hpc params', 'group3 description')
    #hpc stuff
    group3.add_argument('--hpc', action='store_true', help='run script on hpc')
    group3.add_argument('--hpc_walltime', default="72:00:00", help='')
    group3.add_argument('--hpc_nodes', default="nodes=1:ppn=18", help='')

    args = parser.parse_args()

    main(args)
