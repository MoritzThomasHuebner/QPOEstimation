#!/bin/bash
for i in {0..7}
do
  sbtach submit.sh ${i} gaussian_process 5 16
  sbtach submit.sh ${i} periodogram 5 16
done

for i in {0..18}
do
  sbtach submit.sh ${i} gaussian_process 10 40
  sbtach submit.sh ${i} periodogram 10 40
done

for i in {0..11}
do
  sbtach submit.sh ${i} gaussian_process 40 128
  sbtach submit.sh ${i} periodogram 40 128
done
