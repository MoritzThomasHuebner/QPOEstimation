#!/bin/bash

#for segment in {0..7}
#do
#  for period in {0..46}
#  do
#    sbatch analyse_submit.sh giant_flare ${period} ${segment} red_noise  gaussian_process
#    sbatch analyse_submit.sh giant_flare ${period} ${segment} general_qpo  gaussian_process
#    sbatch analyse_submit.sh giant_flare ${period} ${segment} red_noise  gaussian_process_windowed
#    sbatch analyse_submit.sh giant_flare ${period} ${segment} general_qpo  gaussian_process_windowed
#  done
#done

#for period in {0..46}
#do
#  sbatch analyse_submit.sh giant_flare ${period} 6 red_noise  gaussian_process_windowed 2 3.5
#  sbatch analyse_submit.sh giant_flare ${period} 6 qpo  gaussian_process_windowed 2 3.5
#  sbatch analyse_submit.sh giant_flare ${period} 6 general_qpo  gaussian_process_windowed 2 3.5
#  sbatch analyse_submit.sh giant_flare ${period} 27 red_noise  gaussian_process_windowed 1 1.2
#  sbatch analyse_submit.sh giant_flare ${period} 27 qpo  gaussian_process_windowed 1 1.2
#  sbatch analyse_submit.sh giant_flare ${period} 27 general_qpo  gaussian_process_windowed 1 1.2
#done

# GRB
#for n_components in {0..4}
#do
#   for model in red_noise general_qpo
#   do
#     sbatch analyse_submit.sh ${model} ${n_components} fred
#     sbatch analyse_submit.sh ${model} ${n_components} skew_gaussian
#     sbatch analyse_submit.sh ${model} ${n_components} fred_norris
#     sbatch analyse_submit.sh ${model} ${n_components} fred_norris_extended
#   done
#done

# Magnetar flares
#for n_components in {0..2}
#do
#   for model in red_noise qpo pure_qpo general_qpo
#   do
#     sbatch analyse_submit.sh gaussian_process ${model} ${n_components} gaussian
#     sbatch analyse_submit.sh gaussian_process ${model} ${n_components} skew_gaussian
#     sbatch analyse_submit.sh gaussian_process_windowed ${model} ${n_components} gaussian
#     sbatch analyse_submit.sh gaussian_process_windowed ${model} ${n_components} skew_gaussian
#   done
#done

#for injection_id in {0..99}
#do
#  sbatch analyse_submit.sh injection ${injection_id} gaussian_process
#  sbatch analyse_submit.sh injection ${injection_id} gaussian_process_windowed
#done
#

#for likelihood_model in gaussian_process gaussian_process_windowed
#do
#    for recovery_mode in red_noise general_qpo qpo pure_qpo
#    do
#        for background_model in fred gaussian
#        do
#            for n_components in {1..5}
#            do
#                sbatch analyse_submit.sh 121022782 $likelihood_model $recovery_mode $background_model $n_components 950 1500
#                sbatch analyse_submit.sh 120704187 $likelihood_model $recovery_mode $background_model $n_components 350 850
#            done
#        done
#    done
#done

# Hares and hounds
for filename in 10788 122522 129113 159804 161404 165101 166659 172202 173913 181841 186811 189376 19470 19778 202467 206268 236578 265386 265704 270100 271094 294407 294432 296222 297054 299667 300829 302062 305069 317016 319153 325910 329919 337630 338322 346770 352381 36096 361207 37274 379291 379938 384314 389450 389522 395421 400029 40287 404267 419839 428659 42890 429937 455290 458825 46475 472866 48429 487534 495362 539360 542708 555140 562635 563266 5700 598396 609742 612579 65858 673371 705126 71150 711796 732564 744131 758518 761192 762506 773629 820146 821893 845195 846898 857620 864638 869934 870036 878437 889948 898375 908144 909778 912965 923302 935871 940137 968250 976425 997404
do
   for model in red_noise general_qpo
   do
      for n_components in {1}
      do
        sbatch analyse_submit.sh ${filename} gaussian_process ${model} ${n_components} fred_norris_extended
      done
#      for n_components in {1..3}
#      do
#        sbatch analyse_submit.sh ${filename} gaussian_process ${model} ${n_components} gaussian
      done
   done
done