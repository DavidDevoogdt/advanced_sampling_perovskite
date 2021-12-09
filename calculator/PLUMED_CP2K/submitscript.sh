#!/bin/bash
#
#PBS -N TestASEMDCP2K
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=18


module purge

cluster=$(module list | grep -o "cluster/\w*" |  cut -c 9-)
n=$(whoami)

echo $n
echo $cluster

case $n in
  'vsc43693')
    pargs='vsc43693/Libraries'
    case $cluster in
      'swalot')
        s='/user/gent/436/vsc43693/scratch/envs/swalot/bin/activate'
        ;;
      'victini' | 'skitty')  
        s='/user/gent/436/vsc43693/scratch/envs/ASECP2K/bin/activate'
        ;;
      *)
        echo "please add this cluster to script"
        ;;
    esac
    ;;
  'vsc42365')
    pargs='vsc42365/Libraries'
    s='/user/gent/423/vsc42365/scratch/ForInstall/ASECP2K/ASECP2K/bin/activate'
    ;;
  *)
    echo "not added"
    ;;
esac

echo $pargs
echo $s

# When running a Python shell on the login nodes, it is possible that some OpenBLAS-dependent imports
# fail. This is due to a limit on the number of threads that may be used on the login nodes, and
# may be resolved by adding
#
#         import os
#         os.environ['OPENBLAS_NUM_THREADS'] = '1'
#
# before those imports which ensures that the number of created threads does not exceed the limit set by the
# tier 1 administrators


cd ${PBS_O_WORKDIR}

# ACTIVATE ENVIRONMENT
module load Python/3.8.2-GCCcore-9.3.0 
module load CP2K/7.1-intel-2020a
source $s

date
python main.py $pargs
#python main.py 
date





