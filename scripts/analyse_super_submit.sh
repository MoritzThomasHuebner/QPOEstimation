#!/bin/bash

#for segment in {0..27}
for segment in {0..31}
do
  for period in {0..46}
  do
    sbatch analyse_submit.sh $segment $period gaussian_process
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
