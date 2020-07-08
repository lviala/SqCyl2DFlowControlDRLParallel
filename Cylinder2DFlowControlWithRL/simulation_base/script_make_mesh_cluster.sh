#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=8gb

# Cluster Environment Setup
cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate fenicsproject

python3 make_mesh.py

exit 0