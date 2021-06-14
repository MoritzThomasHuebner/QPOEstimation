#!/bin/bash


start_times=(3500 3600 3700 3800 3900 4000 4100 4200 4250 4300 4400 4500 4600 4700 4800 4850 4900 4920 4940 4960 4980)
end_times=(6500 6400 6300 6200 6100 6000 5900 5800 5750 5700 5600 5500 5400 5300 5200 5150 5100 5080 5060 5040 5020)


for recovery_mode in general_qpo red_noise
do
  for i in {0..40}
  do
    sbatch analyse_periodogram_submit.sh ${start_times[$i]} ${end_times[$i]} ${recovery_mode} 0
    sbatch analyse_periodogram_submit.sh ${start_times[$i]} ${end_times[$i]} ${recovery_mode} 1
  done
done
