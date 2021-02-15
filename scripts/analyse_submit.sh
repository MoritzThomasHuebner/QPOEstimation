#!/bin/bash
#
#SBATCH --job-name=qpo
#
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=1G

srun python analyse.py --run_mode injection --injection_id ${1} --injection_mode ${2} --recovery_mode ${2} --model ${3} --plot True --band_minimum 5 --band_maximum 64 --min_log_a -2 --max_log_a 1 --min_log_c -1 --background_model polynomial --polynomial_max 0 --segment_length 1 --nlive 1000 --plot False
