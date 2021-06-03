#!/bin/bash

start_times=(30 20 -4 -25 -50 -100 -150 -200)
end_times=(70 80 103 125 150 200 250 300)

for i in {0..7}
do
  for recovery_mode in qpo no_qpo
  do
    sbatch analyse_grb_periodogram_submit.sh ${start_times[$i]} ${end_times[$i]} $recovery_mode GRB090709A
  done
done


start_times=(-10 -20 -30 -40 -60 -120 -180 -250)
end_times=(10 20 30 40 60 120 180 250)

for i in {0..7}
do
  for recovery_mode in qpo no_qpo
  do
    sbatch analyse_grb_periodogram_submit.sh ${start_times[$i]} ${end_times[$i]} $recovery_mode GRB050128
  done
done
