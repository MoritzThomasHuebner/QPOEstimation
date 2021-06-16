#!/bin/bash
#
#SBATCH --job-name=qpo
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G


srun python analyse_periodogram.py --start_time -${1} --end_time ${1} --data_source injection --injection_id ${2} --injection_mode general_qpo --recovery_mode ${3} --run_mode select_time --injection_file_dir injection_files_pop --injection_likelihood_model whittle --likelihood_model whittle --sample rslice --nlive 300 --try_load True --resume True
