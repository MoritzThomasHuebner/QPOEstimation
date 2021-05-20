import QPOEstimation
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("Qt5Agg")
import pandas as pd

import os

try:
    flares = np.array(sorted(os.listdir('hares_and_hounds_HH2_just_figures')))
except Exception:
    flares = np.array(sorted(os.listdir('hares_and_hounds_HH2')))
print(len(flares))
run_mode = 'from_maximum'

results = np.loadtxt("data/hares_and_hounds/qpp_type_hh2.txt")
flare_keys = results[:, 0]
flare_types = results[:, 1]
flare_keys = [int(k) for k in flare_keys]
flare_types = [int(k) for k in flare_types]

mean_qpo_log_amplitudes = []
for k, t in zip(flare_keys, flare_types):
    try:
        res1 = QPOEstimation.result.GPResult.from_json(f"hares_and_hounds_HH2/{k}/from_maximum/general_qpo/gaussian_process/results/from_maximum_3_gaussians_result.json")
        res2 = QPOEstimation.result.GPResult.from_json(f"hares_and_hounds_HH2/{k}/from_maximum/general_qpo/gaussian_process/results/from_maximum_3_freds_result.json")
        # res3 = QPOEstimation.result.GPResult.from_json(f"hares_and_hounds_HH2/{k}/from_maximum/general_qpo/gaussian_process/results/from_maximum_1_freds_result.json")
        # res4 = QPOEstimation.result.GPResult.from_json(f"hares_and_hounds_HH2/{k}/from_maximum/general_qpo/gaussian_process/results/from_maximum_2_freds_result.json")
        means = [np.mean(res1.posterior['kernel:terms[0]:log_a']), np.mean(res2.posterior['kernel:terms[0]:log_a']),]
                 # np.mean(res3.posterior['kernel:terms[0]:log_a']), np.mean(res4.posterior['kernel:terms[0]:log_a'])]
        mean_qpo_log_amplitudes.append(np.mean(means))
    except Exception as e:
        print(e)
        mean_qpo_log_amplitudes.append(np.nan)
    print(f"{k}\t{t}\t{mean_qpo_log_amplitudes[-1]}")

mean_qpo_log_amplitudes = np.array(mean_qpo_log_amplitudes)
flare_keys = [k for _,k in sorted(zip(mean_qpo_log_amplitudes, flare_keys))]
flare_types = [k for _,k in sorted(zip(mean_qpo_log_amplitudes, flare_types))]
mean_qpo_log_amplitudes = sorted(mean_qpo_log_amplitudes)

for k, m, t in zip(flare_keys, mean_qpo_log_amplitudes, flare_types):
    print(f"{k}\t{t}\t{m}")
