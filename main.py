from genericpath import exists
from loader import main as l
import config
import pathlib
import time
import os
import sys
import argparse

# sets up a new folder for all the output. If run on hpc, the python scripts is called with qsub. Config file is copied as reference


def main(foldername=None, hpc=False, debug=False):

    if foldername is None:
        foldername = time.strftime("%Y-%m-%d_%H-%M-%S")

    rp = pathlib.Path(__file__).parent.resolve()

    data_folder = rp / 'data' / foldername

    if not exists(data_folder):
        data_folder.mkdir(parents=True)
    os.system("cp config.py data/{}/config.py".format(foldername))
    os.chdir(data_folder.resolve())

    if hpc:  # hpc user
        write_submit(rp, debug)
        os.system('qsub submitscript.sh')
        print("job submitted, folder name {}".format(foldername))
    else:  # regular user
        print("folder name {}, running directly".format(foldername))
        l(debug)


def write_submit(rp, debug):
    with open(rp / "calculator" / 'submitscript_template.sh') as f:
        submitscript = f.read().format(config.walltime, config.nodes,
                                       str(rp.resolve()), str(debug))
    with open("submitscript.sh", 'w') as f:
        f.write(submitscript)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulations')

    parser.add_argument(
        '--foldername',
        type=str,
        default=None,
        help='foldername for all output. Default -> current time')
    parser.add_argument('--hpc', action='store_true', help='run script on hpc')
    parser.add_argument('--debug', action='store_true', help='debug')

    args = parser.parse_args()

    main(args.foldername, args.hpc, args.debug)
