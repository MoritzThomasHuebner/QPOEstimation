#!/bin/bash
#
#SBATCH --job-name=qpo
#
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=1G

srun python analyse.py --run_mode ${1} --injection_id ${2} --injection_mode pure_qpo --recovery_mode pure_qpo --amplitude_min 10 --amplitude_max 100 --skewness_min 0.1 --skewness_max 10 --sigma_min 0.1 --sigma_max 1 --t_0_min 0 --t_0_max 1 --min_log_a -1 --max_log_a 1 --min_log_c -1 --max_log_c 1 --likelihood_model gaussian_process --background_model fred --n_components 1 --segment_length 3 --sampling_frequency 256 --band_minimum 1 --band_maximum 64 --plot True --nlive 200 --sample rwalk
