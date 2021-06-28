#!/bin/bash

end_time=20
extensions=($(seq 0 5 100))

sbatch analyse_periodogram_submit.sh -end_time end_time 0 red_noise boxcar
sbatch analyse_periodogram_submit.sh -end_time end_time 1 red_noise tukey
sbatch analyse_periodogram_submit.sh -end_time end_time 0 general_qpo boxcar
sbatch analyse_periodogram_submit.sh -end_time end_time 1 general_qpo tukey


for i in {1..20}
do
  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 0 red_noise hann
  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 1 red_noise hann
  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 0 general_qpo hann
  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 1 general_qpo hann
done

#
#extensions=($(seq 0 10 200))
#start_time=-4
#end_time=103
#
#for i in {1..100}
#do
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) red_noise hann
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) general_qpo hann
#done