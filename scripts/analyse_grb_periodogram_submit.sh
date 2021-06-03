#!/bin/bash
#
#SBATCH --job-name=qpo
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4G

srun python analyse_grb_periodogram.py ${1} ${2} ${3} ${4}
