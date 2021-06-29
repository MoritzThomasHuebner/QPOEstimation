#!/bin/bash
#
#SBATCH --job-name=qpo
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G


# Injection
srun python analyse_periodogram.py --start_time ${1} --end_time ${2} --data_source injection --injection_id ${3} --injection_mode general_qpo --recovery_mode ${4} --window ${5} --frequency_mask_minimum 0.025 --run_mode select_time --injection_file_dir injection_files_pop --injection_likelihood_model whittle --likelihood_model whittle --band_minimum 0.025 --normalisation False --band_maximum 20 --sample rslice --nlive 500 --try_load False --resume False

#GRB
#srun python analyse_periodogram.py --start_time ${1} --end_time ${2} --data_source grb --grb_id 090709A --grb_detector swift --grb_binning 64ms --band_minimum 0.05 --band_maximum 0.30 --recovery_mode ${3} --window ${4} --run_mode select_time --likelihood_model whittle --sample rslice --nlive 1000 --try_load False --resume False
