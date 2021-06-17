#!/bin/bash



times=($(seq 20 20 2000))

for i in {10..101}
do
  sbatch analyse_periodogram_submit.sh ${times[$i]} 0 red_noise
  sbatch analyse_periodogram_submit.sh ${times[$i]} 1 red_noise
  sbatch analyse_periodogram_submit.sh ${times[$i]} 0 general_qpo
  sbatch analyse_periodogram_submit.sh ${times[$i]} 1 general_qpo
done
