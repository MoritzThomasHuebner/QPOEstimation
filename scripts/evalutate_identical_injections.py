import argparse
import sys

import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum_id", default=1000, type=int)
    parser.add_argument("--maximum_id", default=2000, type=int)
    parser.add_argument("--injection_mode", default="qpo", choices=["qpo", "white_noise", "red_noise"], type=str)
    parser.add_argument("--likelihood_model", default="gaussian_process",
                        choices=["gaussian_process", "gaussian_process_windowed"], type=str)
    parser.add_argument("--band_minimum", default=5, type=float)
    parser.add_argument("--band_maximum", default=64, type=float)

    args = parser.parse_args()
    minimum_id = args.minimum_id
    maximum_id = args.maximum_id
    injection_mode = args.injection_mode
    likelihood_model = args.likelihood_model
    band_minimum = args.band_minimum
    band_maximum = args.band_maximum
else:
    matplotlib.use('Qt5Agg')
    minimum_id = 1000
    maximum_id = 2000

    band_minimum = 5
    band_maximum = 64

    injection_mode = "qpo"
    likelihood_model = "gaussian_process"


band = f'{band_minimum}_{band_maximum}Hz'

# averaged runs

log_bfs = []
for injection_id in range(minimum_id, maximum_id):
    res_qpo = bilby.result.read_in_result(f"injection_{band}_normal_{injection_mode}/qpo/results/{injection_id}_{likelihood_model}_result.json")
    res_red_noise = bilby.result.read_in_result(f"injection_{band}_normal_{injection_mode}/red_noise/results/{injection_id}_{likelihood_model}_result.json")
    log_bfs.append(res_qpo.log_evidence - res_red_noise.log_evidence)
    print(log_bfs[-1])
    print(injection_id)

min_ln_bf = np.min(log_bfs)
max_ln_bf = np.max(log_bfs)

plt.hist(log_bfs, bins="fd", label="1000 injections")
plt.axvline(min_ln_bf, label=f"minimum ln BF {min_ln_bf:.2f}", color='orange')
plt.axvline(max_ln_bf, label=f"maximum ln BF {max_ln_bf:.2f}", color='green')
plt.xlabel("ln BF QPO")
plt.ylabel('counts')
plt.legend()
plt.title(f'{injection_mode} injections')
plt.savefig(f'identical_{injection_mode}_injections.png')
plt.show()
