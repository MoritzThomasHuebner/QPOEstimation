#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G


#srun python sliding_window.py --run_id ${1} --period_number ${2} --n_qpos 0 --model gaussian_process --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --run_id ${1} --period_number ${2} --n_qpos 1 --model gaussian_process --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --run_id ${1} --period_number ${2} --n_qpos 0 --model periodogram --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --run_id ${1} --period_number ${2} --n_qpos 1 --model periodogram --band_minimum 10 --band_maximum 40

# Candidate runs
srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 0 --model gaussian_process --plot True --band_minimum 5 --band_maximum 16
srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 1 --model gaussian_process --plot True --band_minimum 5 --band_maximum 16
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 0 --model periodogram --plot True --band_minimum 5 --band_maximum 16
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 1 --model periodogram --plot True --band_minimum 5 --band_maximum 16
#
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 0 --model gaussian_process --plot True --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 1 --model gaussian_process --plot True --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 0 --model periodogram --plot True --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 1 --model periodogram --plot True --band_minimum 10 --band_maximum 40
#
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 0 --model gaussian_process --plot True --band_minimum 40 --band_maximum 128
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 1 --model gaussian_process --plot True --band_minimum 40 --band_maximum 128
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 0 --model periodogram --plot True --band_minimum 40 --band_maximum 128
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --n_qpos 1 --model periodogram --plot True --band_minimum 40 --band_maximum 128
