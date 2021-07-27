#!/bin/bash

#end_time=20
#extensions=($(seq 0 10 480))
#
#sbatch analyse_periodogram_submit.sh -${end_time} ${end_time} 6 red_noise boxcar
#sbatch analyse_periodogram_submit.sh -${end_time} ${end_time} 7 red_noise tukey
#sbatch analyse_periodogram_submit.sh -${end_time} ${end_time} 6 general_qpo boxcar
#sbatch analyse_periodogram_submit.sh -${end_time} ${end_time} 7 general_qpo tukey
#
#
#for i in {1..48}
#do
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 6 red_noise hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 7 red_noise hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 6 general_qpo hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 7 general_qpo hann
#done
#
#for i in {1..45}
#do
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 6 red_noise hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 7 red_noise hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 6 general_qpo hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 7 general_qpo hann
#done

#start_times=(73000 74700 74900 73000)
#end_times=(74700 74900 75800 75800)
#
#for i in {0..3}
#do
##  sbatch analyse_periodogram_submit.sh $((start_times[$i])) $((end_times[$i])) red_noise
##  sbatch analyse_periodogram_submit.sh $((start_times[$i])) $((end_times[$i])) general_qpo
#  sbatch analyse_periodogram_submit.sh $((start_times[$i])) $((end_times[$i])) broken_power_law
#done


#extensions=($(seq 0 10 190))
#start_time=-10
#end_time=10


#sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 00 general_qpo tukey 0.1
#sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 00 red_noise tukey 0.1
#sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 01 general_qpo tukey 0.1
#sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 01 red_noise tukey 0.1
#
#
#for i in {1..19}
#do
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 00 general_qpo hann 0.1
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 00 red_noise hann 0.1
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 01 general_qpo hann 0.1
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 01 red_noise hann 0.1
#done

extensions=($(seq 0 5 90))
start_time=-10
end_time=10


for i in {0..18}
do
  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 02 red_noise hann 0.1
  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 02 general_qpo hann 0.1
done

#for i in {0..19}
#do
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 09 pure_qpo hann 0.025
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 09 white_noise hann 0.025
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 10 pure_qpo hann 0.025
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 10 white_noise hann 0.025
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 11 white_noise tukey 0.5
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 11 pure_qpo tukey 0.5
#done
