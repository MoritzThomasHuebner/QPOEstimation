import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use("Qt5Agg")
from copy import deepcopy
segments = np.arange(0, 38)
mean_log_bfs = []

# for i in range(31):
#     log_bfs = []
#     mean_log_bf = 0
#
#     for j in range(9):
#         try:
#             res_qpo = bilby.result.read_in_result(f"sliding_window/period_{j}/{i}_qpo_result.json")
#             res_no_qpo = bilby.result.read_in_result(f"sliding_window/period_{j}/{i}_no_qpo_result.json")
#             log_bf = res_qpo.log_evidence - res_no_qpo.log_evidence
#             mean_log_bf += log_bf
#             print(f"{j} {i}: {log_bf}")
#         except Exception:
#             pass
#     mean_log_bfs.append(mean_log_bf)
#     print(f"{i}: {mean_log_bf}")
#
# plt.plot(segments, mean_log_bfs)
# plt.show()

import numpy as np

period_one_log_bf_data = []
period_two_log_bf_data = []

for period in range(0, 44):
    log_bfs_one_qpo = []
    log_bfs_two_qpo = []
    mean_frequency = []
    std_frequency = []
    for run_id in range(151):
        try:
            res_no_qpo = bilby.result.read_in_result(f"sliding_window_fine/period_{period}/no_qpo/{run_id}_result.json")
            res_one_qpo = bilby.result.read_in_result(f"sliding_window_fine/period_{period}/one_qpo/{run_id}_result.json")
            # res_two_qpo = bilby.result.read_in_result(f"sliding_window/period_{period}/two_qpo/{run_id}_result.json")
            # res_two_qpo = bilby.result.read_in_result(f"sliding_window/period_{period}/two_qpo/{run_id}_two_qpo_result.json")
            log_bf_one_qpo = res_one_qpo.log_evidence - res_no_qpo.log_evidence
            # log_bf_two_qpo = res_two_qpo.log_evidence - res_no_qpo.log_evidence
            # max_likelihood_sample_one_qpo = res_one_qpo.posterior.iloc[-1]
            # mean_frequency.append(1 / np.exp(max_likelihood_sample_one_qpo[f'kernel:terms[1]:log_P']))
            log_P_samples = np.array(res_one_qpo.posterior['kernel:terms[1]:log_P'])
            frequency_samples = 1 / np.exp(log_P_samples)
            mean_frequency.append(np.mean(frequency_samples))
            std_frequency.append(np.std(frequency_samples))
        except Exception:
            log_bf_one_qpo = np.nan
            # log_bf_two_qpo = np.nan
            mean_frequency.append(np.nan)
            std_frequency.append(np.nan)
        log_bfs_one_qpo.append(log_bf_one_qpo)
        # log_bfs_two_qpo.append(log_bf_two_qpo)
        print(f"{period} {run_id} one qpo: {log_bf_one_qpo}")
        # print(f"{period} {run_id} two qpo: {log_bf_two_qpo}")
    np.savetxt(f'sliding_window_fine/log_bfs_period_one_qpo_{period}', np.array(log_bfs_one_qpo))
    # np.savetxt(f'log_bfs_period_two_qpo_{period}', np.array(log_bfs_two_qpo))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('segment number')
    ax1.set_ylabel('ln BF', color=color)
    ax1.plot(segments, log_bfs_one_qpo, color=color, ls='solid', label='One QPO')
    # ax1.plot(segments, log_bfs_two_qpo, color=color, ls='dotted', label='Two QPOs')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('frequency [Hz]', color=color)  # we already handled the x-label with ax1
    ax2.plot(segments, mean_frequency, color=color)
    mean_frequency = np.array(mean_frequency)
    std_frequency = np.array(std_frequency)
    plt.fill_between(segments, mean_frequency + std_frequency, mean_frequency - std_frequency, color=color, alpha=0.3,
                     edgecolor="none")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend()
    plt.savefig(f'sliding_window_fine/log_bfs_period_{period}')
    plt.clf()
    # period_one_log_bf_data.append(deepcopy(log_bfs_one_qpo))
    # period_two_log_bf_data.append(deepcopy(log_bfs_two_qpo))

# for period in range(9):
#     plt.plot(segments, period_one_log_bf_data[period], label=f'period_{period}')
#     plt.legend()
#     plt.xlabel("Data segment")
#     plt.ylabel("ln BF")
# plt.savefig("one_qpo_log_bfs")
# plt.clf()

# for period in range(9):
#     plt.plot(segments, period_two_log_bf_data[period], label=f'period_{period}')
#     plt.legend()
#     plt.xlabel("Data segment")
#     plt.ylabel("ln BF")
# plt.savefig("two_qpo_log_bfs")
# plt.clf()
#
# for period in range(9):
#     plt.plot(segments, np.array(period_two_log_bf_data[period]) - np.array(period_one_log_bf_data[period]), label=f'period_{period}')
#     plt.legend()
#     plt.xlabel("Data segment")
#     plt.ylabel("ln BF")
# plt.savefig("two_v_one_qpo_log_bfs")
# plt.clf()
