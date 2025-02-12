mkdir inputs
mkdir outputs
cp POSCAR.gz inputs
cp ../tight_relax_1/inputs/POTCAR.* inputs
cp INCAR.gz inputs
cp inputs/* outputs
cp CONTCAR* outputs
cp OUTCAR* outputs
cp vasp* outputs
