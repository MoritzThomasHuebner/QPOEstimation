#!/bin/bash




#for i in {10..100}
#do
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 0 red_noise
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 1 red_noise
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 2 red_noise
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 3 red_noise
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 0 general_qpo
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 1 general_qpo
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 2 general_qpo
#  sbatch analyse_periodogram_submit.sh ${times[$i]} 3 general_qpo
#done


extensions=($(seq 0 2 200))
start_time=5
end_time=20

for i in {0..100}
do
#  echo $((start_time - extensions[$i]))
#  echo $((end_time + extensions[$i]))
  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) red_noise
  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) general_qpo
done