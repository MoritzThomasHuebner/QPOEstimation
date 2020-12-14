#!/bin/bash

#for segment in {0..27}
for segment in 4 13 26
#for segment in 4
#for segment in {0..31}
do
  for period in {0..46}
  do
#    sbatch analyse_submit.sh $segment $period 5 64
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process white_noise
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process qpo
    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process mixed
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process red_noise
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process zeroed_qpo
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process zeroed_mixed
    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed qpo
    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed red_noise
    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed mixed
    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed zeroed_qpo
    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed zeroed_mixed
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process white_noise
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process qpo
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process red_noise
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process mixed
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process zeroed_qpo
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process zeroed_mixed
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process_windowed qpo
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process_windowed red_noise
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process_windowed mixed
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process_windowed zeroed_qpo
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process_windowed zeroed_mixed
#    sbatch analyse_submit.sh $segment $period 128 256 gaussian_process

#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed qpo
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed red_noise
#    sbatch analyse_submit.sh $segment $period 5 64 gaussian_process_windowed mixed

#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process_windowed qpo
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process_windowed red_noise
#    sbatch analyse_submit.sh $segment $period 64 128 gaussian_process_windowed mixed
#    sbatch analyse_submit.sh $segment $period 128 256 gaussian_process_windowed
  done
done

#for injection in {1000..1099}
#do
#  sbatch analyse_submit.sh $injection red_noise
#  sbatch analyse_submit.sh $injection qpo
#done

#for i in {0..44}
#do
#  sbatch analyse_submit.sh ${i}
#done

#for i in {0..999}
#do
#  sbatch analyse_submit.sh ${i} red_noise
#done
