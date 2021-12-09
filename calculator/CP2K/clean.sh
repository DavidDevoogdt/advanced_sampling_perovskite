echo $#

if [ $# -eq 1 ]
then
    mkdir $1
fi

for VAR in bck.* *.e15* *.o15* md.traj cp2k* COLVAR HILLS outplumed WALLS generated.inp TestASEMDCP2K.* core.* md.*
do 
    if [ $# -eq 1 ]
    then
        mv $VAR $1/$VAR
    else
        rm $VAR
    fi
done

if [ $# -eq 1 ]
then
    for VAR in plumed.dat orig_cp2k.inp
    do 
        cp $VAR $1/$VAR
    done
fi


# rm bck.*
# rm *.e15*
# rm *.o15*
# rm md.traj
# rm cp2k*
# rm COLVAR
# rm HILLS    
# rm outplumed   
# rm WALLS
# rm generated.inp
# rm TestASEMDCP2K.*
# rm core.*