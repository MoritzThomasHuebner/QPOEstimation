#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --ntasks=1
#SBATCH --time=15:00
#SBATCH --mem-per-cpu=32G

srun python detrending_data.py ${1}