#!/bin/bash
#
#PBS -N TestASEMDCP2K
#PBS -l walltime={}
#PBS -l {}

cd ${{PBS_O_WORKDIR}}

source activate {}
module load CP2K/7.1-intel-2020a

date
python {}/src/loader.py
date




