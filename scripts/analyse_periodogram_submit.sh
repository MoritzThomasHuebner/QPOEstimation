#!/bin/bash
#
#SBATCH --job-name=qpo
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G


# Injection
#srun python analyse_periodogram.py --start_time ${1} --end_time ${2} --data_source injection --injection_id ${3} --injection_mode qpo_plus_red_noise --recovery_mode ${4} --window ${5} --run_mode select_time --injection_file_dir injection_files_pop --injection_likelihood_model whittle --likelihood_model whittle --frequency_mask_minimum ${6} --band_minimum ${7} --normalisation False --sample rslice --nlive 500 --try_load False --resume False

# Solar Flare
#srun python analyse_periodogram.py --start_time ${1} --end_time ${2} --data_source solar_flare --solar_flare_folder goes --solar_flare_id go1520130512 --recovery_mode ${3} --window hann --run_mode select_time --likelihood_model whittle --normalisation True --sample rwalk --nlive 1000 --try_load False --resume False


#GRB
#srun python analyse_periodogram.py --start_time ${1} --end_time ${2} --data_source grb --grb_id 090709A --grb_detector swift --grb_binning 64ms --band_minimum 0.05 --band_maximum 0.30 --recovery_mode ${3} --window ${4} --run_mode select_time --likelihood_model whittle --sample rslice --nlive 1000 --try_load False --resume False

# Hares and hounds
srun python analyse_periodogram.py --data_source hares_and_hounds --run_mode from_maximum  --hares_and_hounds_id ${1} --hares_and_hounds_round HH2  --likelihood_model whittle --recovery_mode ${2} --sample rwalk --nlive 2000 --use_ratio False --try_load False --resume False --plot True --offset True --jitter_term True --sampling_frequency 1
