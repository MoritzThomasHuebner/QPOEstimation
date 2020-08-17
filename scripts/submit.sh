#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=60:00
#SBATCH --mem-per-cpu=1G

srun python sliding_window.py ${1}