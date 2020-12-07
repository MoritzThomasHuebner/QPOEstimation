#!/bin/bash

#for segment in {0..27}
for segment in 13 26
#for segment in {0..31}
do
  for period in {0..46}
  do
#    sbatch analyse_submit.sh $segment $period 5 64
    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process
    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process
#    sbatch analyse_submit.sh $segment $period 128 256 gaussian_process
    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed
    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process_windowed
#    sbatch analyse_submit.sh $segment $period 128 256 gaussian_process_windowed
  done
done


#for i in {0..44}
#do
#  sbatch analyse_submit.sh ${i}
#done

#for i in {0..999}
#do
#  sbatch analyse_submit.sh ${i} red_noise
#done
