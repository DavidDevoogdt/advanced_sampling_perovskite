from genericpath import exists
from loader import main as l

import numpy
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

    if exists('config.pickle') and not args.override:
        print(
            "using existing config, use --override to force new config in existing folder "
        )

    else:
        pickle.dump(args, open('config.pickle', 'wb'))
        with open('config.txt', 'w') as f:
            f.write(str(args))

    if args.hpc:  # hpc user
        write_submit(rp, args)
        os.system('qsub submitscript.sh')
        print("job submitted, folder name {}".format(args.foldername))
    else:  # regular user
        print("folder name {}, running directly".format(args.foldername))
        l()


def write_submit(rp, args):
    with open(rp / "calculator" / 'submitscript_template.sh') as f:
        submitscript = f.read().format(
            args.hpc_walltime,
            "nodes={}:ppn={}".format(args.hpc_nodes, args.hpc_ppn),
            str(rp.resolve()))
    with open("submitscript.sh", 'w') as f:
        f.write(submitscript)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulations')

    group0 = parser.add_argument_group('general settings', '')

    group0.add_argument('-d', '--debug', action='store_true', help='')
    group0.add_argument('-o',
                        '--override',
                        action='store_true',
                        help='ignore args if folder already exists')

    group1 = parser.add_argument_group('folders and paths', 'default paths')
    group1.add_argument(
        '-n',
        '--foldername',
        type=str,
        default=time.strftime("%Y-%m-%d_%H-%M-%S"),
        help='foldername for all output. Default -> current time')

    group1.add_argument('--pf', default="", help='')

    # no need to specify these:
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

    group3 = parser.add_argument_group('hpc params', '')
    # hpc stuff
    group3.add_argument('--hpc', action='store_true', help='run script on hpc')
    group3.add_argument('--hpc_walltime', default="72:00:00", help='')
    group3.add_argument('--hpc_nodes', type=int, default=1, help='')
    group3.add_argument('--hpc_ppn', type=int, default=9, help='')

    group4 = parser.add_argument_group('Boltzmann generators (bg)',
                                       'parameters to tune bg')

    group4.add_argument('--bg', default=True, help='set to false to skip bg')
    group4.add_argument('--bg_no_train', action='store_true')
    group4.add_argument('--bg_no_path', action='store_true')

    group4.add_argument('--bg_rNVP_layers',
                        type=int,
                        default=11,
                        help='number of realNVP layers')
    group4.add_argument('--bg_NN_layers',
                        type=int,
                        default=2,
                        help='number of hidden layer per densenet')
    group4.add_argument('--bg_NN_nodes',
                        type=int,
                        default=5,
                        help='number of nodes per densenet layer')

    group4.add_argument('--bg_batch_size',
                        type=int,
                        default=1,
                        help='batch size for training')
    group4.add_argument('--bg_nll_rounds', type=int, default=1000, help='')
    group4.add_argument('--bg_kll_rounds',
                        type=int,
                        default=2000,
                        help='number of kll training rounds')
    group4.add_argument('--bg_nll_lr',
                        type=float,
                        default=1e-2,
                        help='learning rate kll training')
    group4.add_argument('--bg_kll_lr',
                        type=float,
                        default=1e-2,
                        help='learning rate kll training')

    group4.add_argument(
        '--bg_n_presample',
        type=int,
        default=5000,
        help='number of monte carlo samples starting from init config')

    args = parser.parse_args()

    main(args)
