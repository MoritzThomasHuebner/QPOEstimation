#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --ntasks=1
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=1G

srun python sliding_window.py ${1} ${2}