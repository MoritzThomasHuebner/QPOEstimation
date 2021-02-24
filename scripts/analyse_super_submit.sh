#!/bin/bash

#for segment in {0..7}
#do
#  for period in {0..46}
#  do
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed red_noise
#  done
#done

for injection_id in {0..99}
do
  sbatch analyse_submit.sh injection ${injection_id}
done

