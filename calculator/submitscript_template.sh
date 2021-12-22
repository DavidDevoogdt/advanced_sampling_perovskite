#!/bin/bash
#
#PBS -N TestASEMDCP2K
#PBS -l walltime={}
#PBS -l {}

. ~/setup_python.sh

cd ${{PBS_O_WORKDIR}}

date
python {}/loader.py 
date




