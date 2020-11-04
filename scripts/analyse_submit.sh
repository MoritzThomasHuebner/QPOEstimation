#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G

srun python analyse.py --run_mode injection --injection_id ${1} --injection_mode ${2} --recovery_mode qpo --model gaussian_process --plot True --band_minimum 10 --band_maximum 64 --min_log_a -2 --max_log_a 1 --min_log_c 1 --polynomial_max 0
srun python analyse.py --run_mode injection --injection_id ${1} --injection_mode ${2} --recovery_mode red_noise --model gaussian_process --plot True --band_minimum 10 --band_maximum 64 --min_log_a -2 --max_log_a 1 --min_log_c 1 --polynomial_max 0
