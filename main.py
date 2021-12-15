from os import mkdir
from src.bg import bg
from src.loader import main as l
import config
import pathlib
import time
import os
import sys

# sets up a new folder for all the output. If run on hpc, the python scripts is called with qsub. Config file is copied as reference


def main():
    rp = pathlib.Path(__file__).parent
    foldername = time.strftime("%Y-%m-%d_%H-%M-%S")
    data_folder = rp/'data'/foldername
    data_folder.mkdir(parents=True)
    os.system("cp config.py data/{}/config.py".format(foldername))
    os.chdir(data_folder.resolve())

    user = os.getlogin()
    if user.startswith('vsc'):  # hpc user
        write_submit(rp)
        os.system('qsub submitscript.sh')
        print("job submitted, folder name {}".format(foldername))
    else:  # regular user
        l()


def write_submit(rp):
    with open(rp/"calculator"/'submitscript_template.sh') as f:
        submitscript = f.read().format(config.walltime, config.nodes,
                                       config.condaenv, str(rp.resolve()))
    with open("submitscript.sh", 'w') as f:
        f.write(submitscript)


if __name__ == "__main__":
    main()
