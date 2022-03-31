#!/bin/bash

# PP Test
#python inject.py --minimum_id 0 --maximum_id 100 --injection_mode qpo_plus_red_noise --amplitude_min 10 --amplitude_max 100 --sigma_min 0.1 --sigma_max 1 --t_0_min 0 --t_0_max 2 --min_log_a -1 --max_log_a 1 --min_log_c_red_noise -1 --max_log_c_red_noise --min_log_c_qpo -1 --max_log_c_qpo 1 --likelihood_model celerite --background_model skew_gaussian --n_components 1 --segment_length 2 --sampling_frequency 256 --band_minimum 1 --band_maximum 64 --plot True

# Model selection low QPO amplitude
#python inject.py --minimum_id ${1} --maximum_id ${2} --injection_mode qpo_plus_red_noise --amplitude_min 3 --amplitude_max 3.00001 --sigma_min 0.2 --sigma_max 0.200001 --t_0_min 0.5 --t_0_max 0.500001 --min_log_a_red_noise 1 --max_log_a_red_noise 1.00001 --min_log_a -2.00001 --max_log_a -2 --min_log_c_red_noise 1 --max_log_c_red_noise 1.00001 --min_log_c_qpo 1 --max_log_c_qpo 1.00001 --likelihood_model celerite --background_model skew_gaussian --n_components 1 --segment_length 1 --sampling_frequency 256 --band_minimum 20 --band_maximum 20 --plot True

# Model selection high QPO amplitude
python inject.py --minimum_id ${1} --maximum_id ${2} --injection_mode qpo_plus_red_noise --amplitude_min 3 --amplitude_max 3.00001 --sigma_min 0.2 --sigma_max 0.200001 --t_0_min 0.5 --t_0_max 0.500001 --min_log_a_red_noise 1 --max_log_a_red_noise 1.00001 --min_log_a -0.6 --max_log_a -0.59999 --min_log_c_red_noise 1 --max_log_c_red_noise 1.00001 --min_log_c_qpo 1 --max_log_c_qpo 1.00001 --likelihood_model celerite --background_model skew_gaussian --n_components 1 --segment_length 1 --sampling_frequency 256 --band_minimum 20 --band_maximum 20 --plot False
