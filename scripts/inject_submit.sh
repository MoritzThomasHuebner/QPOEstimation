#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G

srun python inject.py --minimum_id ${1} --maximum_id  ${2} --injection_mode ${3}
