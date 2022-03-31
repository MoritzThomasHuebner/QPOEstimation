#!/bin/bash

# Hares and hounds
#for filename in 10788 122522 129113 159804 161404 165101 166659 172202 173913 181841 186811 189376 19470 19778 202467 206268 236578 265386 265704 270100 271094 294407 294432 296222 297054 299667 300829 302062 305069 317016 319153 325910 329919 337630 338322 346770 352381 36096 361207 37274 379291 379938 384314 389450 389522 395421 400029 40287 404267 419839 428659 42890 429937 455290 458825 46475 472866 48429 487534 495362 539360 542708 555140 562635 563266 5700 598396 609742 612579 65858 673371 705126 71150 711796 732564 744131 758518 761192 762506 773629 820146 821893 845195 846898 857620 864638 869934 870036 878437 889948 898375 908144 909778 912965 923302 935871 940137 968250 976425 997404
#do
#   for model in red_noise qpo_plus_red_noise
#   do
#      for n_components in {0..2}
#      do
#        sbatch analyse_submit.sh ${filename} celerite ${model} ${n_components} skew_exponential
#      done
#   done
#done


### GRB
#for model in red_noise qpo_plus_red_noise
#do
#  for n_components in {1..4}
#  do
#    sbatch analyse_submit.sh celerite ${model} skew_exponential ${n_components}
#    sbatch analyse_submit.sh celerite ${model} fred ${n_components}
#    sbatch analyse_submit.sh celerite ${model} fred_extended ${n_components}
#    sbatch analyse_submit.sh celerite ${model} skew_gaussian ${n_components}
#  done
#done

## Magnetar flare
#for model in red_noise qpo_plus_red_noise
#do
#  for n_components in {1..3}
#  do
#    sbatch analyse_submit.sh celerite ${model} skew_exponential ${n_components}
#    sbatch analyse_submit.sh celerite ${model} fred ${n_components}
#    sbatch analyse_submit.sh celerite ${model} fred_extended ${n_components}
#    sbatch analyse_submit.sh celerite ${model} skew_gaussian ${n_components}
#  done
#done

#for n_components in {3..10}
#do
#  sbatch analyse_submit.sh white_noise fred ${n_components}
#done


#start_times=(4900 4920 4940 4960 4980)
#end_times=(5100 5080 5060 5040 5020)
#start_times=(0 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 4100 4200 4250 4300 4400 4500 4600 4700 4800 4850 4900 4920 4940 4960 4980)
#end_times=(10000 8000 7900 7800 7700 7600 7500 7400 7300 7200 7100 7000 6900 6800 6700 6600 6500 6400 6300 6200 6100 6000 5900 5800 5750 5700 5600 5500 5400 5300 5200 5150 5100 5080 5060 5040 5020)
#
#recovery_mode="red_noise"
#
#for i in {0..50}
#do
#  sbatch analyse_submit.sh ${start_times[$i]} ${end_times[$i]} ${recovery_mode} 0 celerite
#  sbatch analyse_submit.sh ${start_times[$i]} ${end_times[$i]} ${recovery_mode} 0 celerite_windowed
#done

# POP injections

#extensions=($(seq 0 10 190))
#start_time=-10
#end_time=10
#
#
#for i in {0..19}
#do
#  sbatch analyse_submit.sh 01 red_noise celerite $((start_time - extensions[$i])) $((end_time + extensions[$i]))
#  sbatch analyse_submit.sh 01 red_noise celerite_windowed $((start_time - extensions[$i])) $((end_time + extensions[$i]))
#  sbatch analyse_submit.sh 01 qpo_plus_red_noise celerite $((start_time - extensions[$i])) $((end_time + extensions[$i]))
#  sbatch analyse_submit.sh 01 qpo_plus_red_noise celerite_windowed $((start_time - extensions[$i])) $((end_time + extensions[$i]))
#done

# MSS test

for injection_mode in qpo_plus_red_noise #red_noise
do
    for recovery_mode in red_noise qpo_plus_red_noise
    do
        for injection_id in {1000..1999}
        do
          sbatch analyse_submit.sh $injection_id $injection_mode $recovery_mode
        done
    done
done

## PP

#for recovery_mode in red_noise qpo_plus_red_noise
#do
#    for injection_id in {0..100}
#    do
#      sbatch analyse_submit.sh $injection_id $recovery_mode $recovery_mode
#    done
#done


### Giant flare

#period=7.56
#start_time_base=103.0
#end_time_base=106.0
#
#for i in {1..4}
#do
#  sbatch analyse_submit.sh red_noise celerite 121.06 123.06 ${i}
#  sbatch analyse_submit.sh red_noise celerite_windowed 121.06 123.06 ${i}
#  sbatch analyse_submit.sh qpo_plus_red_noise celerite 121.06 123.06 ${i}
#  sbatch analyse_submit.sh qpo_plus_red_noise celerite_windowed 121.06 123.06 ${i}
#done

#period=7.56
#start_time_base=98.38
#end_time_base=100.38
#
#for i in {0..20}
#do
#  start_time=$(python -c "print(${start_time_base} + ${period} * ${i})")
#  end_time=$(python -c "print(${end_time_base} + ${period} * ${i})")
#  sbatch analyse_submit.sh red_noise celerite_windowed ${start_time} ${end_time} 2
#  sbatch analyse_submit.sh qpo_plus_red_noise celerite_windowed ${start_time} ${end_time} 2
#done


## Solar flare

#for i in {0..3}
#do
#  sbatch analyse_submit.sh red_noise ${i}
#  sbatch analyse_submit.sh qpo_plus_red_noise ${i}
#done
