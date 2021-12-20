#!/bin/bash
#
#PBS -N TestASEMDCP2K
#PBS -l walltime={}
#PBS -l {}

. ~/setup_python.sh

cd ${{PBS_O_WORKDIR}}
#module load CP2K/7.1-intel-2020a

# module list
# which python

date
python {}/loader.py 
date




