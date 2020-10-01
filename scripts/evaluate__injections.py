import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use("Qt5Agg")
from copy import deepcopy
import json
n_injections = 100
injections = np.arange(0, n_injections)
# band = '16_32Hz'
# band = 'below_16Hz'
band = '5_64Hz'

log_bfs_one_qpo_gpr = []
log_bfs_one_qpo_whittle = []
log_bfs_one_qpo_poisson = []

perc_unc_gpr = []
perc_unc_whittle = []
perc_unc_poisson = []

for injection in range(n_injections):
    try:
        res_no_qpo = bilby.result.read_in_result(f"sliding_window_{band}_injections_fixed_log_b/no_qpo/results/{str(injection).zfill(2)}_result.json")
        res_one_qpo = bilby.result.read_in_result(f"sliding_window_{band}_injections_free_log_b/one_qpo/results/{str(injection).zfill(2)}_result.json")
        res_no_qpo_whittle = bilby.result.read_in_result(f"sliding_window_{band}_injections_fixed_log_b/no_qpo/results/{str(injection).zfill(2)}_whittle_result.json")
        res_one_qpo_whittle = bilby.result.read_in_result(f"sliding_window_{band}_injections_fixed_log_b/one_qpo/results/{str(injection).zfill(2)}_whittle_result.json")
        res_no_qpo_poisson = bilby.result.read_in_result(f"sliding_window_{band}_injections_fixed_log_b/no_qpo/results/{str(injection).zfill(2)}_poisson_result.json")
        res_one_qpo_poisson = bilby.result.read_in_result(f"sliding_window_{band}_injections_fixed_log_b/one_qpo/results/{str(injection).zfill(2)}_poisson_result.json")
        # res_two_qpo = bilby.result.read_in_result(f"sliding_window_{band}_candidates/two_qpo/results/{candidate}_result.json")
        log_bfs_one_qpo_gpr.append(res_one_qpo.log_evidence - res_no_qpo.log_evidence)
        log_bfs_one_qpo_whittle.append(res_one_qpo_whittle.log_evidence - res_no_qpo_whittle.log_evidence)
        log_bfs_one_qpo_poisson.append(res_one_qpo_poisson.log_evidence - res_no_qpo_poisson.log_evidence)
        # log_bfs_two_qpo.append(res_two_qpo.log_evidence - res_no_qpo.log_evidence)
        # data = np.loadtxt(f"injection_files/{str(injection).zfill(2)}_data.txt")
        # plt.plot(data[:, 0], data[:, 1])
        # plt.xlabel('time [s]')
        # plt.ylabel('counts')
        # plt.savefig(f'injection_files/{str(injection).zfill(2)}_plot')
        # plt.clf()

        with open(f'injection_files/{str(injection).zfill(2)}_params.json', 'r') as f:
            injection_params = json.load(f)
            true_frequency = injection_params['frequency']

        log_P_samples = np.array(res_one_qpo.posterior['kernel:log_P'])
        frequency_samples_gpr = 1 / np.exp(log_P_samples)
        frequency_samples_whittle = np.array(res_one_qpo_whittle.posterior['central_frequency'])
        frequency_samples_poisson = np.array(res_one_qpo_poisson.posterior['frequency'])

        # perc_unc_gpr.append(np.std(frequency_samples_gpr)/np.mean(frequency_samples_gpr))
        # perc_unc_whittle.append(np.std(frequency_samples_whittle)/np.mean(frequency_samples_whittle))
        # perc_unc_poisson.append(np.std(frequency_samples_poisson)/np.mean(frequency_samples_poisson))
        #
        #
        # plt.hist(frequency_samples_gpr, bins="fd", density=True, alpha=0.3, label='gpr')
        # plt.hist(frequency_samples_whittle, bins="fd", density=True, alpha=0.3, label='whittle')
        # plt.hist(frequency_samples_poisson, bins="fd", density=True, alpha=0.3, label='poisson')
        # plt.axvline(true_frequency, label='injected frequency')
        # plt.legend()
        # plt.xlabel('frequency [Hz]')
        # plt.ylabel('probability')
        # plt.savefig(f"sliding_window_{band}_injections/frequency_posterior_{str(injection).zfill(2)}")
        # plt.clf()

    except Exception as e:
        print(e)
        log_bfs_one_qpo_gpr.append(np.nan)
        log_bfs_one_qpo_whittle.append(np.nan)
        log_bfs_one_qpo_poisson.append(np.nan)
# np.savetxt(f'sliding_window_{band}_injections/log_bfs_one_qpo', np.array(log_bfs_one_qpo_gpr))
# np.savetxt(f'sliding_window_{band}_candidates/log_bfs_two_qpo', np.array(log_bfs_two_qpo))

print(f"ID:\tln BF GPR\tln BF Whittle\tln BF Poissson")
for i, log_bf_gpr, log_bf_whittle, log_bf_poisson in zip(np.arange(0, len(log_bfs_one_qpo_gpr)), log_bfs_one_qpo_gpr, log_bfs_one_qpo_whittle, log_bfs_one_qpo_poisson):
    print(f"{i}:\t{log_bf_gpr:.2f}\t{log_bf_whittle:.2f}\t{log_bf_poisson:.2f}")

print(f"Combined gpr ln BF: {np.sum(np.nan_to_num(log_bfs_one_qpo_gpr))}")
print(f"Combined whittle ln BF: {np.sum(np.nan_to_num(log_bfs_one_qpo_whittle))}")
print(f"Combined poisson ln BF: {np.sum(np.nan_to_num(log_bfs_one_qpo_poisson))}")


print(f"ID:\tGPR\tWhittle\tPoissson")
for i, gpr, whittle, poisson in zip(np.arange(0, len(perc_unc_gpr)), perc_unc_gpr, perc_unc_whittle, perc_unc_poisson):
    print(f"{i}:\t{gpr:.2f}\t{whittle:.2f}\t{poisson:.2f}")


print(f"Mean gpr rel. freq. err.: {np.mean(np.nan_to_num(perc_unc_gpr))}")
print(f"Mean whittle rel. freq. err.: {np.mean(np.nan_to_num(perc_unc_whittle))}")
print(f"Mean poisson rel. freq. err.: {np.mean(np.nan_to_num(perc_unc_poisson))}")
