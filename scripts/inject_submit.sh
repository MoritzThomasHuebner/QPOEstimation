#!/bin/bash
#
#SBATCH --job-name=inject
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G

srun python inject.py --minimum_id ${1} --maximum_id  ${2} --injection_mode ${3} --polynomial_max 0 --min_log_a 0 --max_log_a 0 --min_log_c 0 --max_log_c 0 --band_minimum 20 --band_maximum 20
