import QPOEstimation
import numpy as np

from QPOEstimation.result import power_qpo, power_red_noise

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("Qt5Agg")
# import pandas as pd

import os

try:
    flares = np.array(sorted(os.listdir("results/hares_and_hounds_HH2_just_figures")))
except Exception:
    flares = np.array(sorted(os.listdir("results/hares_and_hounds_HH2")))
print(len(flares))
run_mode = "from_maximum"

results = np.loadtxt("data/hares_and_hounds/qpp_type_hh2.txt")
flare_keys = results[:, 0]
flare_types = results[:, 1]
flare_keys = [int(k) for k in flare_keys]
flare_types = [int(k) for k in flare_types]

mean_qpo_log_amplitudes = []

for k, t in zip(flare_keys, flare_types):
    try:
        res1 = QPOEstimation.result.GPResult.from_json(f"results/hares_and_hounds_HH2/{k}/from_maximum/qpo_plus_red_noise/celerite/results/from_maximum_3_gaussians_result.json")
        res2 = QPOEstimation.result.GPResult.from_json(f"results/hares_and_hounds_HH2/{k}/from_maximum/qpo_plus_red_noise/celerite/results/from_maximum_3_skew_exponentials_result.json")
        # res1.plot_qpo_log_amplitude()
        # res1.plot_amplitude_ratio()
        # res2.plot_qpo_log_amplitude()
        # res2.plot_amplitude_ratio()
        res_1_a_qpo = np.exp(res1.posterior["kernel:terms[0]:log_a"])
        res_1_c_qpo = np.exp(res1.posterior["kernel:terms[0]:log_c"])
        res_1_f_qpo = np.exp(res1.posterior["kernel:terms[0]:log_f"])

        res_1_a_red_noise = np.exp(res1.posterior["kernel:terms[1]:log_a"])
        res_1_c_red_noise = np.exp(res1.posterior["kernel:terms[1]:log_c"])

        res_1_power_qpo = np.array(power_qpo(res_1_a_qpo, res_1_c_qpo, res_1_f_qpo))
        res_1_power_red_noise = np.array(power_red_noise(res_1_a_red_noise, res_1_c_red_noise))

        # plt.hist(np.log(res_1_power_qpo), bins="fd", histtype="step")
        # plt.savefig(f"temp_plots/{k}_qpo_power.png")
        # plt.clf()

        # plt.hist(np.log(res_1_power_red_noise), bins="fd", histtype="step")
        # plt.savefig(f"temp_plots/{k}_red_noise_power.png")
        # plt.clf()

        # log_power = np.log(res_1_power_qpo/res_1_power_red_noise)
        # plt.hist(log_power, bins="fd", histtype="step")
        # plt.savefig(f"temp_plots/{k}_power_ratio.png")
        # plt.clf()
        # prior_range = res1.priors["kernel:terms[0]:log_f"].maximum - res1.priors["kernel:terms[0]:log_f"].minimum
        # means = [np.std(res1.posterior["kernel:terms[0]:log_f"])/prior_range]
        res_2_a_qpo = np.exp(res2.posterior["kernel:terms[0]:log_a"])
        res_2_c_qpo = np.exp(res2.posterior["kernel:terms[0]:log_c"])
        res_2_f_qpo = np.exp(res2.posterior["kernel:terms[0]:log_f"])
        res_2_a_red_noise = np.exp(res2.posterior["kernel:terms[1]:log_a"])
        res_2_c_red_noise = np.exp(res2.posterior["kernel:terms[1]:log_c"])
        res_2_power_qpo = power_qpo(res_2_a_qpo, res_2_c_qpo, res_2_f_qpo)
        res_2_power_red_noise = power_red_noise(res_2_a_red_noise, res_2_c_red_noise)

        # res3 = QPOEstimation.result.GPResult.from_json(f"hares_and_hounds_HH2/{k}/from_maximum/qpo_plus_red_noise/celerite/results/from_maximum_1_skew_exponentials_result.json")
        # res4 = QPOEstimation.result.GPResult.from_json(f"hares_and_hounds_HH2/{k}/from_maximum/qpo_plus_red_noise/celerite/results/from_maximum_2_skew_exponentials_result.json")

        means = [np.mean(np.log(res_1_power_qpo)), np.mean(np.log(res_2_power_qpo))]
                 # np.mean(res3.posterior["kernel:terms[0]:log_a"]), np.mean(res4.posterior["kernel:terms[0]:log_a"])]
        mean_qpo_log_amplitudes.append(np.mean(means))
    except Exception as e:
        print(e)
        mean_qpo_log_amplitudes.append(np.nan)
    print(f"{k}\t{t}\t{mean_qpo_log_amplitudes[-1]}")

print()

mean_qpo_log_amplitudes = np.array(mean_qpo_log_amplitudes)
flare_keys = [k for _,k in sorted(zip(mean_qpo_log_amplitudes, flare_keys))]
flare_types = [k for _,k in sorted(zip(mean_qpo_log_amplitudes, flare_types))]
mean_qpo_log_amplitudes = sorted(mean_qpo_log_amplitudes)

for k, m, t in zip(flare_keys, mean_qpo_log_amplitudes, flare_types):
    print(f"{k}\t{t}\t{m}")
