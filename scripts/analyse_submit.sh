#!/bin/bash
#
#SBATCH --job-name=qpo
#
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=1G

#srun python analyse.py --data_source ${1} --injection_id ${2} --injection_mode pure_qpo --recovery_mode pure_qpo --amplitude_min 10 --amplitude_max 100 --skewness_min 0.1 --skewness_max 10 --sigma_min 0.1 --sigma_max 1 --t_0_min 0 --t_0_max 3 --min_log_a -1 --max_log_a 1 --min_log_c -1 --max_log_c 1 --likelihood_model ${3} --background_model fred --n_components 1 --segment_length 3 --sampling_frequency 256 --band_minimum 1 --band_maximum 64 --plot True --nlive 500 --sample rwalk
srun python analyse.py --data_source ${1} --period_number ${2} --run_id ${3} --recovery_mode ${4}  --likelihood_model ${5} --data_mode normal --band_minimum 5 --band_maximum 64 --variance_stabilisation True --background_model polynomial --polynomial_max 1000 --segment_length 1.0 --segment_step 0.945 --sampling_frequency 256 --plot True --nlive 500 --sample rwalk
