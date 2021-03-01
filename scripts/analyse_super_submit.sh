#!/bin/bash

for segment in {0..7}
do
  for period in {13..14}
  do
    sbatch analyse_submit.sh giant_flare ${period} ${segment} red_noise  gaussian_process
    sbatch analyse_submit.sh giant_flare ${period} ${segment} general_qpo  gaussian_process
    sbatch analyse_submit.sh giant_flare ${period} ${segment} red_noise  gaussian_process_windowed
    sbatch analyse_submit.sh giant_flare ${period} ${segment} general_qpo  gaussian_process_windowed
  done
done

#for injection_id in {0..99}
#do
#  sbatch analyse_submit.sh injection ${injection_id} gaussian_process
#  sbatch analyse_submit.sh injection ${injection_id} gaussian_process_windowed
#done
#
