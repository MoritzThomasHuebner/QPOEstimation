#!/bin/bash
#
#SBATCH --job-name=qpo
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G

srun python analyse.py --run_mode injection --injection_id ${1} --injection_mode red_noise --recovery_mode red_noise --model ${2} --plot True --band_minimum 5 --band_maximum 64 --min_log_a -2 --max_log_a 1 --min_log_c -1 --background_model polynomial --polynomial_max 0 --segment_length 1 --nlive 200
srun python analyse.py --run_mode injection --injection_id ${1} --injection_mode red_noise --recovery_mode qpo --model ${2} --plot True --band_minimum 5 --band_maximum 64 --min_log_a -2 --max_log_a 1 --min_log_c -1 --background_model polynomial --polynomial_max 0 --segment_length 1 --nlive 200
srun python analyse.py --run_mode injection --injection_id ${1} --injection_mode qpo --recovery_mode red_noise --model ${2} --plot True --band_minimum 5 --band_maximum 64 --min_log_a -2 --max_log_a 1 --min_log_c -1 --background_model polynomial --polynomial_max 0 --segment_length 1 --nlive 200
srun python analyse.py --run_mode injection --injection_id ${1} --injection_mode qpo --recovery_mode qpo --model ${2} --plot True --band_minimum 5 --band_maximum 64 --min_log_a -2 --max_log_a 1 --min_log_c -1 --background_model polynomial --polynomial_max 0 --segment_length 1 --nlive 200
#srun python analyse.py --run_mode injection --injection_id ${1} --injection_mode qpo --recovery_mode qpo --model ${2} --plot True --band_minimum 5 --band_maximum 64 --min_log_a -2 --max_log_a 1 --min_log_c -1 --minimum_window_spacing 0.5 --background_model polynomial --polynomial_max 10 --segment_length 1 --nlive 500
#srun python analyse.py --run_mode injection --injection_id ${1} --injection_mode qpo --recovery_mode qpo --model gaussian_process --plot True --band_minimum 5 --band_maximum 64 --min_log_a -2 --max_log_a 1 --min_log_c -1 --minimum_window_spacing 0.5 --background_model polynomial --polynomial_max 10 --segment_length 1 --nlive 500
#srun python analyse.py --run_mode injection --injection_id ${1} --injection_mode ${2} --recovery_mode red_noise --model gaussian_process --plot True --band_minimum 5 --band_maximum 64 --min_log_a -2 --max_log_a 4 --min_log_c -4 --background_model mean

#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode white_noise  --model gaussian_process --band_minimum ${3} --band_maximum ${4} --background_model mean --data_mode smoothed_residual --segment_length 1.8 --segment_step 0.23625 --plot True --try_load True
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode ${6}          --model ${5}             --band_minimum ${3} --band_maximum ${4} --background_model polynomial --data_mode normal --segment_length 2.0 --segment_step 0.23625 --plot True --try_load False --use_ratio False --nlive 1000
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode ${6}          --model ${5}             --band_minimum ${3} --band_maximum ${4} --background_model mean --data_mode smoothed_residual --segment_length 2.8 --segment_step 0.945 --plot True --try_load False --use_ratio False

#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode qpo          --model ${5}             --band_minimum ${3} --band_maximum ${4} --background_model polynomial --data_mode normal --segment_length 1.8 --segment_step 0.23625 --plot True --try_load True --use_ratio True
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode red_noise    --model ${5}             --band_minimum ${3} --band_maximum ${4} --background_model polynomial --data_mode normal --segment_length 1.8 --segment_step 0.23625 --plot True --try_load True --use_ratio True
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode mixed        --model ${5}             --band_minimum ${3} --band_maximum ${4} --background_model polynomial --data_mode normal --segment_length 1.8 --segment_step 0.23625 --plot True --try_load True --use_ratio True

#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode qpo          --model gaussian_process_windowed --band_minimum ${3} --band_maximum ${4} --background_model mean --data_mode blind_injection --segment_length 2.0 --segment_step 0.23625 --plot False
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode red_noise    --model gaussian_process_windowed --band_minimum ${3} --band_maximum ${4} --background_model mean --data_mode smoothed_residual --segment_length 1.8 --segment_step 0.23625 --plot True
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode mixed        --model gaussian_process_windowed --band_minimum ${3} --band_maximum ${4} --background_model mean --data_mode smoothed_residual --segment_length 1.8 --segment_step 0.23625 --plot True
#
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode white_noise --model gaussian_process --band_minimum 64 --band_maximum 128 --background_model mean --data_mode smoothed_residual --segment_length 2 --segment_step 0.23625 --plot False --try_load True --use_ratio False
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode qpo --model gaussian_process_windowed --band_minimum 64 --band_maximum 128 --background_model mean --data_mode smoothed_residual --segment_length 2 --segment_step 0.23625 --plot False --try_load True --use_ratio False
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode red_noise --model gaussian_process_windowed --band_minimum 64 --band_maximum 128 --background_model mean --data_mode smoothed_residual --segment_length 2 --segment_step 0.23625 --plot False --try_load True --use_ratio False
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode mixed --model gaussian_process_windowed --band_minimum 64 --band_maximum 128 --background_model mean --data_mode smoothed_residual --segment_length 2 --segment_step 0.23625 --plot False --try_load True --use_ratio False
#
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode white_noise --model gaussian_process --band_minimum 128 --band_maximum 256 --background_model mean --data_mode smoothed_residual --segment_length 2 --segment_step 0.23625 --plot False --try_load True --use_ratio False
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode qpo --model gaussian_process_windowed --band_minimum 128 --band_maximum 256 --background_model mean --data_mode smoothed_residual --segment_length 2 --segment_step 0.23625 --plot False --try_load True --use_ratio False
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode red_noise --model gaussian_process_windowed --band_minimum 128 --band_maximum 256 --background_model mean --data_mode smoothed_residual --segment_length 2 --segment_step 0.23625 --plot False --try_load True --use_ratio False
#srun python analyse.py  --run_id ${1} --period_number ${2} --recovery_mode mixed --model gaussian_process_windowed --band_minimum 128 --band_maximum 256 --background_model mean --data_mode smoothed_residual --segment_length 2 --segment_step 0.23625 --plot False --try_load True --use_ratio False

#srun python analyse.py  --run_mode candidates --candidate_id ${1} --recovery_mode red_noise --model gaussian_process --band_minimum 5 --band_maximum 64 --background_model polynomial --polynomial_max 1000 --data_mode normal --segment_length 1 --plot True --try_load True --resume True
#srun python analyse.py  --run_mode candidates --candidate_id ${1} --recovery_mode qpo --model gaussian_process --band_minimum 5 --band_maximum 64 --background_model polynomial --polynomial_max 1000 --data_mode normal --segment_length 1 --plot True --try_load True --resume True
