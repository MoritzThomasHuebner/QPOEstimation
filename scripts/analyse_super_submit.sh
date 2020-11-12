#!/bin/bash

#for segment in {0..27}
#for segment in {0..7}
#do
#  for period in {0..46}
#  do
#    sbatch analyse_submit.sh $segment $period gaussian_process 64 128
#  done
#done


for i in {0..44}
do
  sbatch analyse_submit.sh ${i}
done

#for i in {0..999}
#do
#  sbatch analyse_submit.sh ${i} qpo
#done
