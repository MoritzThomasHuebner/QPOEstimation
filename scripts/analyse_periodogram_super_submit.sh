#!/bin/bash



times=($(seq 20 20 2000))

for i in {0..9}
do
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 0 red_noise
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 1 red_noise
  sbatch analyse_periodogram_submit.sh ${times[$i]} 2 red_noise
  sbatch analyse_periodogram_submit.sh ${times[$i]} 3 red_noise
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 0 general_qpo
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 1 general_qpo
  sbatch analyse_periodogram_submit.sh ${times[$i]} 2 general_qpo
  sbatch analyse_periodogram_submit.sh ${times[$i]} 3 general_qpo
done
