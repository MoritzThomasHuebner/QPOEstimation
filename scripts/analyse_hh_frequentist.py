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

results = np.loadtxt("data/hares_and_hounds/qpp_type_hh2.txt", dtype=int)
flare_keys = results[:, 0]
flare_types = results[:, 1]
print(flare_keys)
print(flare_types)

mean_qpo_log_amplitudes = []
for k, t in zip(flare_keys, flare_types):
    try:
        res = QPOEstimation.result.GPResult.from_json(f"hares_and_hounds_HH2/{k}/from_maximum/general_qpo/gaussian_process/results/from_maximum_1_gaussians_result.json")
        mean_qpo_log_amplitudes.append(np.mean(res.posterior['kernel:terms[0]:log_a']))
    except Exception as e:
        print(e)
        mean_qpo_log_amplitudes.append(np.nan)
    print(f"{k}\t{t}\t{mean_qpo_log_amplitudes[-1]}")

mean_qpo_log_amplitudes = np.array(mean_qpo_log_amplitudes)
flare_keys = [k for _,k in sorted(zip(mean_qpo_log_amplitudes, flare_keys))]
mean_qpo_log_amplitudes = sorted(mean_qpo_log_amplitudes)

for k, m, t in zip(flare_keys, mean_qpo_log_amplitudes, flare_types):
    print(f"{k}\t{t}\t{m}")
