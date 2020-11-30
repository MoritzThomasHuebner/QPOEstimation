#!/bin/bash

#for segment in {0..27}
for segment in {0..31}
do
  for period in 13 26
  do
    sbatch analyse_submit.sh $segment $period 5 64
    sbatch analyse_submit.sh $segment $period 64 128
    sbatch analyse_submit.sh $segment $period 128 256
  done
done


#for i in {0..44}
#do
#  sbatch analyse_submit.sh ${i}
#done

#for i in {0..999}
#do
#  sbatch analyse_submit.sh ${i} red_noise
#done
