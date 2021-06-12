#!/bin/bash


start_times=(0 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 4100 4200 4250 4300 4400 4500 4600 4700 4800 4850 4900 4920 4940 4960 4980)
end_times=(10000 8000 7900 7800 7700 7600 7500 7400 7300 7200 7100 7000 6900 6800 6700 6600 6500 6400 6300 6200 6100 6000 5900 5800 5750 5700 5600 5500 5400 5300 5200 5150 5100 5080 5060 5040 5020)


for recovery_mode in general_qpo red_noise
do
  for i in {0..70}
  do
    sbatch analyse_periodogram_submit.sh ${start_times[$i]} ${end_times[$i]} ${recovery_mode} 0
    sbatch analyse_periodogram_submit.sh ${start_times[$i]} ${end_times[$i]} ${recovery_mode} 1
  done
done
