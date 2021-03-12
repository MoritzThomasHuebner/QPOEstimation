#!/bin/bash
#
#SBATCH --job-name=qpo
#
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=4G

#srun python analyse.py --data_source ${1} --injection_id ${2} --injection_mode pure_qpo --recovery_mode pure_qpo --amplitude_min 10 --amplitude_max 100 --skewness_min 0.1 --skewness_max 10 --sigma_min 0.1 --sigma_max 1 --t_0_min 0 --t_0_max 3 --min_log_a -1 --max_log_a 1 --min_log_c -1 --max_log_c 1 --likelihood_model ${3} --background_model fred --n_components 1 --segment_length 3 --sampling_frequency 256 --band_minimum 1 --band_maximum 64 --plot True --nlive 500 --sample rwalk
#srun python analyse.py --data_source grb --run_mode select_time--likelihood_model ${5} --data_mode normal --band_minimum 5 --band_maximum 64 --variance_stabilisation False --background_model gaussian --n_components ${6} --polynomial_max 1000 --segment_length ${7} --segment_step 0.2365 --sampling_frequency 256 --plot True --nlive 1000 --sample rslice --amplitude_min 0.001 --amplitude_max 1000 --offset True --offset_min 0 --offset_max 100 --try_load True

#srun python analyse.py --data_source solar_flare --solar_flare_id ${1} --run_mode select_time --likelihood_model ${2} --recovery_mode ${3} --band_minimum 0.001 --band_maximum 1 --variance_stabilisation False --background_model ${4} --n_components ${5} --plot True --nlive 500 --sample rwalk --start_time ${6} --end_time ${7}  --amplitude_min 10 --amplitude_max 1e8 --skewness_min 0.1 --skewness_max 10000 --sigma_min 0.1 --sigma_max 1000 --t_0_min 0 --t_0_max 2000 --min_log_a -30 --max_log_a 30 --suffix "${5}_${4}_" --nlive 5000 --sample rslice

srun python analyse.py --data_source grb --run_mode select_time --grb_id 090709A --grb_binning 1s --start_time -4 --end_time 103 --t_0_min -24 --t_0_max 123 --min_log_a -10 --max_log_a 5 --min_log_c -10 --max_log_c 3 --recovery_mode ${1} --likelihood_model gaussian_process --background_model fred --n_components ${2} --band_minimum 0.02 --band_maximum 0.5 --sample rwalk --nlive 3000 --use_ratio False --try_load False --resume False --plot True --offset False
