#!/bin/bash

#for segment in {0..7}
#do
#  for period in {0..46}
#  do
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed red_noise
#  done
#done

for injection in {0..99}
do
#  sbatch analyse_submit.sh ${injection} qpo gaussian_process
  sbatch analyse_submit.sh ${injection} red_noise gaussian_process
#  sbatch analyse_submit.sh ${injection} general_qpo gaussian_process
done

for injection in {100..199}
do
#  sbatch analyse_submit.sh ${injection} qpo gaussian_process_windowed
  sbatch analyse_submit.sh ${injection} red_noise gaussian_process_windowed
#  sbatch analyse_submit.sh ${injection} general_qpo gaussian_process_windowed
done