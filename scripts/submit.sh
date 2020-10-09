#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G


srun python sliding_window.py --run_id ${1} --period_number ${2} --n_qpos 0 --model gaussian_process --band_minimum 10 --band_maximum 40
srun python sliding_window.py --run_id ${1} --period_number ${2} --n_qpos 1 --model gaussian_process --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --run_id ${1} --period_number ${2} --n_qpos 0 --model periodogram --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --run_id ${1} --period_number ${2} --n_qpos 1 --model periodogram --band_minimum 10 --band_maximum 40

# Candidate runs
#srun python sliding_window.py --candidate_run True --candidate_id ${2} --n_qpos 0 --model gaussian_process --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --candidate_run True --candidate_id ${2} --n_qpos 1 --model gaussian_process --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --candidate_run True --candidate_id ${2} --n_qpos 0 --model periodogram --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --candidate_run True --candidate_id ${2} --n_qpos 1 --model periodogram --band_minimum 10 --band_maximum 40
