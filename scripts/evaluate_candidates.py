import bilby
import numpy as np
import matplotlib
import json
import matplotlib.pyplot as plt
from astropy.io import fits
matplotlib.use('Qt5Agg')


log_bfs_whittle = []
log_bfs_gpr = []
log_bfs_gpr_sho = []
log_bfs_gpr_two_qpo = []
log_bfs_poisson = []

# band = '5_64Hz'
band = 'miller'

bfs_miller = np.array([1.8e7, 2.47e6, 9.77e5, 5.09e5, 2.3e5, 2.09e5, 1.97e5, 1.68e5, 1.6e5, 6.28e4, 2.21e4, 1.1e4, 9.76e3,
                       5.63e3, 4.57e4, 4.3e3, 3.09e3, 2.49e3, 1.83e3, 1.48e3, 1.36e3, 1.32e3, 1.02e3])

for i in range(23):
    # res_no_qpo_whittle = bilby.result.read_in_result(f'sliding_window_{band}_candidates_ref/no_qpo_exponential/results/{i}_whittle_result.json')
    # res_no_qpo_poisson = bilby.result.read_in_result(f'sliding_window_{band}_candidates_ref/no_qpo_exponential/results/{i}_poisson_result.json')
    res_one_qpo_gpr = bilby.result.read_in_result(f'sliding_window_{band}_candidates/one_qpo/results/{i}_result.json')
    res_one_qpo_whittle = bilby.result.read_in_result(f'sliding_window_{band}_candidates_ref/one_qpo/results/{i}_whittle_result.json')
    # res_one_qpo_poisson = bilby.result.read_in_result(f'sliding_window_{band}_candidates_ref/one_qpo/results/{i}_poisson_result.json')
    res_no_qpo_gpr = bilby.result.read_in_result(f'sliding_window_{band}_candidates/no_qpo/results/{i}_result.json')
    res_no_qpo_whittle = bilby.result.read_in_result(f'sliding_window_{band}_candidates_ref/no_qpo/results/{i}_whittle_result.json')
    # res_no_qpo_poisson = bilby.result.read_in_result(f'sliding_window_{band}_candidates_ref/no_qpo/results/{i}_poisson_result.json')
    # res_no_qpo_whittle = bilby.result.read_in_result(f'sliding_window_{band}_injections_fixed_log_b/no_qpo/results/{str(i).zfill(2)}_whittle_result.json')
    # res_one_qpo_whittle = bilby.result.read_in_result(f'sliding_window_{band}_injections_fixed_log_b/one_qpo/results/{str(i).zfill(2)}_whittle_result.json')
    # res_no_qpo_gpr = bilby.result.read_in_result(f'sliding_window_{band}_injections_fixed_log_b/no_qpo/results/{str(i).zfill(2)}_result.json')
    # res_no_qpo_gpr_sho = bilby.result.read_in_result(f'sliding_window_{band}_injections/no_qpo/results/{str(i).zfill(2)}_result.json')
    # res_one_qpo_gpr = bilby.result.read_in_result(f'sliding_window_{band}_injections_fixed_log_b/one_qpo/results/{str(i).zfill(2)}_result.json')
    # res_no_qpo_poisson = bilby.result.read_in_result(f'sliding_window_{band}_injections_fixed_log_b/no_qpo/results/{str(i).zfill(2)}_poisson_result.json')
    # res_one_qpo_poisson = bilby.result.read_in_result(f'sliding_window_{band}_injections_fixed_log_b/one_qpo/results/{str(i).zfill(2)}_poisson_result.json')
    log_bfs_gpr.append(res_one_qpo_gpr.log_evidence - res_no_qpo_gpr.log_evidence)
    # log_bfs_gpr_sho.append(res_one_qpo_gpr.log_evidence - res_no_qpo_gpr_sho.log_evidence)
    log_bfs_whittle.append(res_one_qpo_whittle.log_evidence - res_no_qpo_whittle.log_evidence)
    # log_bfs_poisson.append(res_one_qpo_poisson.log_evidence - res_no_qpo_poisson.log_evidence)
    # log_bfs_whittle.append(np.nan)
    # log_bfs_gpr_two_qpo.append(np.nan)
    # log_bfs_poisson.append(res_one_qpo_poisson.log_bayes_factor)

    # frequency_samples = res_one_qpo_gpr.posterior["frequency"]
    # plt.hist(res_one_qpo_poisson.posterior["f"], bins="fd", density=True, alpha=0.4, label='Poisson')
    # plt.axvline(params['frequency'])
    # plt.hist(1/np.exp(res_one_qpo_gpr.posterior["kernel:log_P"]), bins="fd", density=True, alpha=0.4, label='GPR')
    # plt.hist(res_one_qpo_whittle.posterior["central_frequency"], bins="fd", density=True, alpha=0.4, label='Whittle')
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('normalised PDF')
    # plt.legend()
    # plt.savefig(f"sliding_window_{band}_candidates/frequency_posterior_{str(i).zfill(2)}.png")
    # plt.clf()
    #
    # except Exception as e:
    #     log_bfs_whittle.append(np.nan)
    #     log_bfs_gpr.append(np.nan)
    #     log_bfs_gpr_sho.append(np.nan)
    #     log_bfs_poisson.append(np.nan)
    #     print(e)

# print(log_bfs_gpr)

print(f"ID:\tln BF GPR\tln BF Whittle\tln BF Miller")
for i, log_bf_gpr, log_bf_whittle, log_bf_miller in zip(np.arange(0, len(log_bfs_gpr)), log_bfs_gpr, log_bfs_whittle, np.log(bfs_miller)):
    print(f"{i}:\t{log_bf_gpr:.2f}\t{log_bf_whittle:.2f}\t{log_bf_miller:.2f}")

print(f"Total GP: {np.sum(np.nan_to_num(log_bfs_gpr))}")
print(f"Total Whittle: {np.sum(np.nan_to_num(log_bfs_whittle))}")
print(f"Total Miller: {np.sum(np.nan_to_num(np.log(bfs_miller)))}")
# print(f"ID:\tln BF GPR\tln BF Whittle\tln BF Poisson\t")
# for i, log_bf_gpr, log_bf_gpr_two_qpo, log_bf_whittle, log_bf_poisson in zip(np.arange(0, len(log_bfs_gpr)), log_bfs_gpr, log_bfs_gpr_two_qpo, log_bfs_whittle, log_bfs_poisson):
#     print(f"{i}:\t{log_bf_gpr:.2f}\t{log_bf_gpr_two_qpo:.2f}\t{log_bf_whittle:.2f}\t{log_bf_poisson:.2f}\t")

# np.savetxt(f"sliding_window_{band}_candidates/log_bfs_one_qpo_whittle", log_bfs_whittle)

# ttes = data[:, 0]
#
# frequency = 32
# resolution = 1/frequency
#
# times = np.arange(0, ttes[-1], resolution)
# counts = [0]
#
# for i in range(1, len(times)):
#     counts.append(len(np.where(np.logical_and(ttes > times[i - 1], ttes < times[i]))[0]))
#
# binned_data = np.array([times, counts]).T
# # np.savetxt(f'data/sgr1806_{frequency}Hz.dat', binned_data)
# plt.plot(times, counts)
# plt.show()
#

# p = bilby.core.prior.LogUniform(minimum=0.1, maximum=200)
# ts = np.linspace(0.1, 200, 1000)
#
# plt.plot(ts, p.ln_prob(ts))
# plt.show()