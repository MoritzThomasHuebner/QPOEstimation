#!/bin/bash

start_times=(0 2500 3500 4000 4250 4500 4700 4800 4850 4900 4920 4940 4960 4980)
end_times=(10000 7500 6500 6000 5750 5500 5300 5200 5150 5100 5080 5060 5040 5020)

for recovery_mode in general_qpo red_noise
do
  for i in {0..13}
  do
    sbatch analyse_periodogram_submit.sh ${start_times[$i]} ${end_times[$i]} ${recovery_mode}
  done
done
