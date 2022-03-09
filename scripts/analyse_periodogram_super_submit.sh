#!/bin/bash

# Hares and hounds
for filename in 10788 122522 129113 159804 161404 165101 166659 172202 173913 181841 186811 189376 19470 19778 202467 206268 236578 265386 265704 270100 271094 294407 294432 296222 297054 299667 300829 302062 305069 317016 319153 325910 329919 337630 338322 346770 352381 36096 361207 37274 379291 379938 384314 389450 389522 395421 400029 40287 404267 419839 428659 42890 429937 455290 458825 46475 472866 48429 487534 495362 539360 542708 555140 562635 563266 5700 598396 609742 612579 65858 673371 705126 71150 711796 732564 744131 758518 761192 762506 773629 820146 821893 845195 846898 857620 864638 869934 870036 878437 889948 898375 908144 909778 912965 923302 935871 940137 968250 976425 997404
do
   for model in red_noise qpo_plus_red_noise
   do
      sbatch analyse_periodogram_submit.sh ${filename} ${model}
   done
done


#end_time=20
#extensions=($(seq 0 10 480))
#
#sbatch analyse_periodogram_submit.sh -${end_time} ${end_time} 6 red_noise boxcar
#sbatch analyse_periodogram_submit.sh -${end_time} ${end_time} 7 red_noise tukey
#sbatch analyse_periodogram_submit.sh -${end_time} ${end_time} 6 qpo_plus_red_noise boxcar
#sbatch analyse_periodogram_submit.sh -${end_time} ${end_time} 7 qpo_plus_red_noise tukey
#
#
#for i in {1..48}
#do
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 6 red_noise hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 7 red_noise hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 6 qpo_plus_red_noise hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 7 qpo_plus_red_noise hann
#done
#
#for i in {1..45}
#do
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 6 red_noise hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 7 red_noise hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 6 qpo_plus_red_noise hann
#  sbatch analyse_periodogram_submit.sh -$((end_time + extensions[$i])) $((end_time + extensions[$i])) 7 qpo_plus_red_noise hann
#done

#start_times=(73000 74700 74900 73000)
#end_times=(74700 74900 75800 75800)
#
#for i in {0..3}
#do
##  sbatch analyse_periodogram_submit.sh $((start_times[$i])) $((end_times[$i])) red_noise
##  sbatch analyse_periodogram_submit.sh $((start_times[$i])) $((end_times[$i])) qpo_plus_red_noise
#  sbatch analyse_periodogram_submit.sh $((start_times[$i])) $((end_times[$i])) broken_power_law
#done






#extensions=($(seq 0 10 190))
#start_time=-10
#end_time=10
#
#
#sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 00 qpo_plus_red_noise tukey 0.1 0.5
#sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 00 red_noise tukey 0.1 0.5
#sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 01 qpo_plus_red_noise tukey 0.1 0.5
#sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 01 red_noise tukey 0.1 0.5
#
#for i in {1..19}
#do
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 00 qpo_plus_red_noise hann 0.1 0.5
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 00 red_noise hann 0.1 0.5
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 01 qpo_plus_red_noise hann 0.1 0.5
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 01 red_noise hann 0.1 0.5
#done
#
#extensions=($(seq 0 5 90))
#start_time=-10
#end_time=10
#
#
#for i in {0..18}
#do
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 02 red_noise hann 0.1 0.1
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 02 qpo_plus_red_noise hann 0.1 0.1
#done
#
extensions=($(seq 0 10 190))
start_time=-10
end_time=10

sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 03 pure_qpo tukey 0.1 0.1
sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 03 white_noise tukey 0.1 0.1
#sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 04 pure_qpo hann 0.1 0.1
#sbatch analyse_periodogram_submit.sh $((start_time - extensions[0])) $((end_time + extensions[0])) 04 white_noise hann 0.1 0.1
#
#for i in {0..19}
#do
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 03 pure_qpo hann 0.1 0.1
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 03 white_noise hann 0.1 0.1
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 04 pure_qpo hann 0.1 0.1
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 04 white_noise hann 0.1 0.1
#done
#
#extensions=($(seq 0 10 190))
#start_time=-10
#end_time=10
#
#for i in {0..19}
#do
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 05 white_noise hann 0.5 0.5
#  sbatch analyse_periodogram_submit.sh $((start_time - extensions[$i])) $((end_time + extensions[$i])) 05 pure_qpo hann 0.5 0.5
#done
#
#
