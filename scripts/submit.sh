#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G


#srun python sliding_window.py --run_id ${1} --period_number ${2} --recovery_mode no_qpo --model ${3} --band_minimum ${4} --band_maximum ${5}
#srun python sliding_window.py --run_id ${1} --period_number ${2} --recovery_mode one_qpo --model ${3} --band_minimum ${4} --band_maximum ${5}
#srun python sliding_window.py --run_id ${1} --period_number ${2} --recovery_mode no_qpo --model periodogram --band_minimum 10 --band_maximum 40
#srun python sliding_window.py --run_id ${1} --period_number ${2} --recovery_mode one_qpo --model periodogram --band_minimum 10 --band_maximum 40
#
#srun python sliding_window.py --run_id ${1} --period_number ${2} --recovery_mode no_qpo --model ${3} --band_minimum ${4} --band_maximum ${5}
#srun python sliding_window.py --run_id ${1} --period_number ${2} --recovery_mode one_qpo --model ${3} --band_minimum ${4} --band_maximum ${5}
#srun python sliding_window.py --run_id ${1} --period_number ${2} --recovery_mode no_qpo --model periodogram --band_minimum 10 --band_maximum 40  --suffix _0.5s --segment_length 0.5
#srun python sliding_window.py --run_id ${1} --period_number ${2} --recovery_mode one_qpo --model periodogram --band_minimum 10 --band_maximum 40  --suffix _0.5s --segment_length 0.5
#
# Candidate runs
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --recovery_mode no_qpo --model ${2} --plot True --band_minimum ${3} --band_maximum ${4}
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --recovery_mode one_qpo --model ${2} --plot True --band_minimum ${3} --band_maximum ${4}
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --recovery_mode no_qpo --model ${2} --plot True --band_minimum ${3} --band_maximum ${4} --suffix _0.5s --segment_length 0.5
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --recovery_mode one_qpo --model ${2} --plot True --band_minimum ${3} --band_maximum ${4} --suffix _0.5s --segment_length 0.5
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --recovery_mode no_qpo --model periodogram --plot True --band_minimum 5 --band_maximum 16
#srun python sliding_window.py --candidates_run True --candidate_id ${1} --recovery_mode one_qpo --model periodogram --plot True --band_minimum 5 --band_maximum 16
#
#srun python sliding_window.py --candidates_run True --miller_candidates True --candidate_id ${1} --recovery_mode no_qpo --model gaussian_process --plot True
#srun python sliding_window.py --candidates_run True --miller_candidates True --candidate_id ${1} --recovery_mode one_qpo --model gaussian_process --plot True
#srun python sliding_window.py --candidates_run True --miller_candidates True --candidate_id ${1} --recovery_mode no_qpo --model periodogram --plot True
#srun python sliding_window.py --candidates_run True --miller_candidates True --candidate_id ${1} --recovery_mode one_qpo --model periodogram --plot True
#
srun python sliding_window.py --injection_run True --injection_id ${1} --injection_mode ${2} --recovery_mode no_qpo --model gaussian_process --plot True --band_minimum 5 --band_maximum 64
srun python sliding_window.py --injection_run True --injection_id ${1} --injection_mode ${2} --recovery_mode one_qpo --model gaussian_process --plot True --band_minimum 5 --band_maximum 64
srun python sliding_window.py --injection_run True --injection_id ${1} --injection_mode ${2} --recovery_mode red_noise --model gaussian_process --plot True --band_minimum 5 --band_maximum 64
#srun python sliding_window.py --injection_run True --injection_id ${1} --injection_mode red_noise --recovery_mode no_qpo --model gaussian_process --plot True --band_minimum 5 --band_maximum 64 --suffix _exp_kernel
#srun python sliding_window.py --injection_run True --injection_id ${1} --injection_mode red_noise --recovery_mode one_qpo --model gaussian_process --plot True --band_minimum 5 --band_maximum 64 --suffix _exp_kernel
#srun python sliding_window.py --injection_run True --injection_id ${1} --injection_mode red_noise --recovery_mode red_noise --model gaussian_process --plot True --band_minimum 5 --band_maximum 64 --suffix _exp_kernel
