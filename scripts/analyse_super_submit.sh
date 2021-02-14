#!/bin/bash

#for segment in {0..7}
#do
#  for period in {0..46}
#  do
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed red_noise
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed zeroed_mixed
#  done
#done

for injection in {0..99}
do
  sbatch analyse_submit.sh ${injection} gaussian_process
done