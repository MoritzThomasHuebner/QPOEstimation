#!/bin/bash
#
#SBATCH --job-name=qpo
#
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=2G


# GRB
#srun python analyse.py --data_source grb --run_mode select_time --grb_id 090709A --grb_binning 1s --grb_detector swift --start_time -4 --end_time 103  --likelihood_model ${1} --recovery_mode ${2} --background_model ${3} --n_components ${4} --sample rwalk --nlive 2000 --resume True --try_load True --plot True --offset False --jitter_term False

# Magnetar flare
#srun python analyse.py --data_source magnetar_flare_binned --run_mode entire_segment --magnetar_label SGR_0501 --magnetar_tag 080823478_lcobs --rebin_factor 8 --likelihood_model ${1} --recovery_mode ${2} --background_model ${3} --n_components ${4} --plot True --nlive 2000 --sample rwalk --resume True --try_load True --offset False --jitter_term False

# Hares and Hounds
#srun python analyse.py --data_source hares_and_hounds --run_mode from_maximum  --hares_and_hounds_id ${1} --hares_and_hounds_round HH2  --likelihood_model ${2} --recovery_mode ${3} --n_components ${4} --background_model ${5} --sample rwalk --nlive 1000 --use_ratio False --try_load True --resume True --plot True --offset True --jitter_term True --sampling_frequency 1

# Injection
#srun python analyse.py --start_time ${1} --end_time ${2} --data_source injection --injection_id ${4} --injection_mode qpo_plus_red_noise --recovery_mode ${3} --run_mode select_time --injection_file_dir injection_files_pop --injection_likelihood_model celerite_windowed --likelihood_model ${5} --sample rwalk --nlive 500 --try_load True --resume True --background_model 0
#srun python analyse.py --data_source injection --run_mode select_time --injection_id ${1} --injection_mode qpo_plus_red_noise --injection_file_dir injection_files_pop --injection_likelihood_model whittle --recovery_mode ${2} --likelihood_model ${3} --background_model 0 --plot True --nlive 500 --sample rslice --start_time ${4} --end_time ${5} --offset False --jitter_term True --min_log_a -5 --max_log_a 10
#srun python analyse.py --data_source injection --run_mode entire_segment --injection_id ${1} --injection_mode ${2} --base_injection_outdir injections/injection_pp_non_eq_dis --injection_file_dir injection_files_pp_non_eq_dis --injection_likelihood_model celerite --recovery_mode ${3} --plot True --nlive 1500 --sample rwalk --offset False --amplitude_min 10 --amplitude_max 100 --sigma_min 0.1 --sigma_max 1 --t_0_min 0 --t_0_max 2 --min_log_a -1 --max_log_a 1 --min_log_c_red_noise -1 --max_log_c_red_noise 1 --min_log_c_qpo -1 --max_log_c_qpo 1 --likelihood_model celerite --background_model skew_gaussian --n_components 1 --sampling_frequency 256 --band_minimum 1 --band_maximum 64 --plot True --try_load False
#srun python analyse.py --data_source injection --run_mode entire_segment --injection_id ${1} --injection_mode ${2} --base_injection_outdir injections/injection --injection_file_dir injection_files_pop --injection_likelihood_model ${3} --recovery_mode ${2} --plot True --nlive 500 --sample rslice --offset False --min_log_a -2 --max_log_a 2 --min_log_c_red_noise -1 --max_log_c_red_noise 3 --min_log_c_qpo -1 --max_log_c_qpo 3 --likelihood_model ${3} --background_model skew_gaussian --n_components 1 --sampling_frequency 256 --band_minimum 1 --band_maximum 64 --plot True
#srun python analyse.py --data_source injection --run_mode entire_segment --injection_id ${1} --injection_mode ${2} --base_injection_outdir injections/injection_mss --injection_file_dir injection_files_mss --injection_likelihood_model celerite --recovery_mode ${3} --likelihood_model celerite --plot True --nlive 500 --sample rslice --offset False --min_log_a -2 --max_log_a 2 --min_log_c_red_noise -1 --max_log_c_red_noise --min_log_c_qpo -1 --max_log_c_qpo 3 --likelihood_model celerite --background_model 0 --sampling_frequency 256 --band_minimum 1 --band_maximum 64 --plot True
srun python analyse.py --data_source injection --run_mode entire_segment --injection_id ${1} --injection_mode ${2} --base_injection_outdir results/injection_files_mss_with_mean --injection_file_dir injections/injection_files_mss_with_mean --injection_likelihood_model celerite --recovery_mode ${3} --likelihood_model celerite --plot False --nlive 500 --sample rslice --offset False --min_log_a -4 --max_log_a 4 --min_log_c_red_noise -1 --max_log_c_red_noise 3 --min_log_c_qpo -1 --max_log_c_qpo 3 --amplitude_min 0.1 --amplitude_max 10 --likelihood_model celerite --background_model skew_gaussian --n_components 1 --sampling_frequency 256 --band_minimum 1 --band_maximum 64 --plot False

# Giant Flare
#srun python analyse.py --data_source giant_flare --run_mode select_time  --recovery_mode ${1} --likelihood_model ${2} --start_time ${3} --end_time ${4} --background_model skew_gaussian --n_components ${5} --offset True --plot True --nlive 1000 --sample rslice --resume True --try_load True --sampling_frequency 64

# Solar Flare
#srun python analyse.py --start_time 74700 --end_time 74900 --data_source solar_flare --solar_flare_folder goes --solar_flare_id go1520130512 --recovery_mode ${1} --background_model polynomial --run_mode select_time --likelihood_model celerite --min_log_a -12 --normalisation True --sample rwalk --nlive 2000 --n_components ${2} --jitter_term False --try_load True --resume True
