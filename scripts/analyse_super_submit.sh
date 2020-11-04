#!/bin/bash

#for segment in {0..27}
#do
#  for period in {0..46}
#  do
#    sbatch submit.sh $segment $period gaussian_process 64 128
#  done
#done


for i in {0..99}
do
  sbatch analyse_submit.sh ${i} red_noise
  sbatch analyse_submit.sh ${i} red_noise
  sbatch analyse_submit.sh ${i} red_noise
done

for i in {0..999}
do
  sbatch analyse_submit.sh ${i} qpo
  sbatch analyse_submit.sh ${i} qpo
  sbatch analyse_submit.sh ${i} qpo
done
