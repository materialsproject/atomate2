#!/bin/bash

#SBATCH --partition=micro
#SBATCH --job-name=dft_phonon_static_eos_deformation_1
#SBATCH --nodes=3
#SBATCH --ntasks=144
#SBATCH --time=02:55:00
#SBATCH --account=pn73da
#SBATCH --mail-user=your_email@adress
#SBATCH --mail-type=ALL
#SBATCH --output=/hppfs/scratch/00/di82tut/autoplex_test/run/8a/70/ab/8a70ab31-57c5-4b5d-a2f3-1d23e48fc4a1_1/queue.out
#SBATCH --error=/hppfs/scratch/00/di82tut/autoplex_test/run/8a/70/ab/8a70ab31-57c5-4b5d-a2f3-1d23e48fc4a1_1/queue.err
#SBATCH --get-user-env
cd /hppfs/scratch/00/di82tut/autoplex_test/run/8a/70/ab/8a70ab31-57c5-4b5d-a2f3-1d23e48fc4a1_1
export ATOMATE2_CONFIG_FILE="/dss/dsshome1/00/di82tut/.atomate2/config/atomate2.yaml"
source activate autoplex_test
module load slurm_setup
module load vasp/6.1.2

jf -fe execution run /hppfs/scratch/00/di82tut/autoplex_test/run/8a/70/ab/8a70ab31-57c5-4b5d-a2f3-1d23e48fc4a1_1
