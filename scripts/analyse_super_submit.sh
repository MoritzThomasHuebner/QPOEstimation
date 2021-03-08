#!/bin/bash

#for segment in {0..7}
#do
#  for period in {0..46}
#  do
#    sbatch analyse_submit.sh giant_flare ${period} ${segment} red_noise  gaussian_process
#    sbatch analyse_submit.sh giant_flare ${period} ${segment} general_qpo  gaussian_process
#    sbatch analyse_submit.sh giant_flare ${period} ${segment} red_noise  gaussian_process_windowed
#    sbatch analyse_submit.sh giant_flare ${period} ${segment} general_qpo  gaussian_process_windowed
#  done
#done

for period in {0..46}
do
  sbatch analyse_submit.sh giant_flare ${period} 10 red_noise  gaussian_process_windowed 2 2.5
  sbatch analyse_submit.sh giant_flare ${period} 10 qpo  gaussian_process_windowed 2 2.5
done

for period in {0..46}
do
  sbatch analyse_submit.sh giant_flare ${period} 27 red_noise  gaussian_process_windowed 1 1.2
  sbatch analyse_submit.sh giant_flare ${period} 27 qpo  gaussian_process_windowed 1 1.2
done


#for injection_id in {0..99}
#do
#  sbatch analyse_submit.sh injection ${injection_id} gaussian_process
#  sbatch analyse_submit.sh injection ${injection_id} gaussian_process_windowed
#done
#

#for likelihood_model in gaussian_process gaussian_process_windowed
#do
#    for recovery_mode in red_noise general_qpo qpo pure_qpo
#    do
#        for background_model in fred gaussian
#        do
#            for n_components in {1..5}
#            do
#                sbatch analyse_submit.sh 121022782 $likelihood_model $recovery_mode $background_model $n_components 950 1500
#                sbatch analyse_submit.sh 120704187 $likelihood_model $recovery_mode $background_model $n_components 350 850
#            done
#        done
#    done
#done