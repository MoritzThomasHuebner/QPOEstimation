#!/bin/bash

#start_times=(0 2500 3500 4000 4250 4500 4700 4800 4850 4900 4920 4940 4960 4980)
#end_times=(10000 7500 6500 6000 5750 5500 5300 5200 5150 5100 5080 5060 5040 5020)
start_times=(2000 2100 2200 2300 2400 2600 2700 2800 2900 3000 3100 3200 3300 3400 3600 3700 3800 3900 4100 4200 4300 4400 4600)
end_times=(8000 7900 7800 7700 7600 7400 7300 7200 7100 7000 6900 6800 6700 6600 6400 6300 6200 6100 5900 5800 5700 5600 5400)

for recovery_mode in general_qpo red_noise
do
  for i in {0..13}
  do
    sbatch analyse_periodogram_submit.sh ${start_times[$i]} ${end_times[$i]} ${recovery_mode}
  done
done