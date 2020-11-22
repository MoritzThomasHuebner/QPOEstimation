import bilby
import numpy as np
import matplotlib
import json
import matplotlib.pyplot as plt
from astropy.io import fits
matplotlib.use('Qt5Agg')


log_bfs = []

band = '5_64Hz'
# band = 'miller'

bfs_miller = np.loadtxt('candidates/miller_bayes_factors.dat')
data_mode = 'smoothed_residual'
likelihood_model = 'gaussian_process'


for i in range(45):
    try:
        res_qpo_gpr = bilby.result.read_in_result(f'candidates_{band}_{data_mode}/qpo/results/{i}_{likelihood_model}_result.json')
        res_red_noise_gpr = bilby.result.read_in_result(f'candidates_{band}_{data_mode}/red_noise/results/{i}_{likelihood_model}_result.json')
        log_bfs.append(res_qpo_gpr.log_evidence - res_red_noise_gpr.log_evidence)
    except OSError:
        continue

# print(log_bfs_gpr)

print(f"ID:\tln BF GPR")
for i, log_bf_gpr in zip(np.arange(0, len(log_bfs)), log_bfs):
    print(f"{i}:\t{log_bf_gpr:.2f}")

print(f"Total QPO: {np.sum(np.nan_to_num(log_bfs))}")
