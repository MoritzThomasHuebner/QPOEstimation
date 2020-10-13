#!/bin/bash
for i in {0..7}
do
  sbatch submit.sh ${i} gaussian_process 5 16
  sbatch submit.sh ${i} periodogram 5 16
done

for i in {0..18}
do
  sbatch submit.sh ${i} gaussian_process 10 40
  sbatch submit.sh ${i} periodogram 10 40
done

for i in {0..11}
do
  sbatch submit.sh ${i} gaussian_process 40 128
  sbatch submit.sh ${i} periodogram 40 128
done
