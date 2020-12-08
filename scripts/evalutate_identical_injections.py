import argparse
import sys

import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

band_minimum = 5
band_maximum = 64

likelihood_model = 'gaussian_process'

band = f'{band_minimum}_{band_maximum}Hz'

log_bfs_one_qpo_gpr = []
log_bfs_one_qpo_whittle = []


# averaged runs

log_a = 1
log_c = -1
log_f = np.log(20)

log_evidences_qpo = []
log_evidences_red_noise = []
averaged_log_bfs_qpo_v_red_noise = []
averaged_log_bfs_qpo_v_red_noise_err = []

injection_mode = "qpo"
log_bfs = []
for injection_id in range(1000, 1100):
    res_qpo = bilby.result.read_in_result(f"injection_{band}_normal_{injection_mode}/qpo/results/{injection_id}_{likelihood_model}_result.json")
    res_red_noise = bilby.result.read_in_result(f"injection_{band}_normal_{injection_mode}/red_noise/results/{injection_id}_{likelihood_model}_result.json")
    log_bfs.append(res_qpo.log_evidence - res_red_noise.log_evidence)
    print(log_bfs[-1])
    print(injection_id)

min_ln_bf = np.min(log_bfs)
max_ln_bf = np.max(log_bfs)

plt.hist(log_bfs, bins=50, label="100 injections")
plt.axvline(min_ln_bf, label=f"minimum ln BF {min_ln_bf:.2f}", color='orange')
plt.axvline(max_ln_bf, label=f"maximum ln BF {max_ln_bf:.2f}", color='green')
plt.xlabel("ln BF QPO")
plt.ylabel('counts')
plt.legend()
plt.title(f'{injection_mode} injections')
plt.savefig(f'identical_{injection_mode}_injections.png')
plt.show()
