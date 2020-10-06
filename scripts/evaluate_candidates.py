import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use("Qt5Agg")
from copy import deepcopy
n_candidates = 26
candidates = np.arange(0, n_candidates)
# band = '16_32Hz'
# band = 'below_16Hz'
band = '32_16Hz'

for candidate in range(len(candidates)):
    log_bfs_one_qpo = []
    log_bfs_two_qpo = []
    try:
        # res_no_qpo = bilby.result.read_in_result(f"sliding_window_{band}_candidates/no_qpo/results/{candidate}_result.json")
        res_one_qpo = bilby.result.read_in_result(f"sliding_window_{band}_candidates/one_qpo/results/{candidate}_result.json")
        # res_two_qpo = bilby.result.read_in_result(f"sliding_window_{band}_candidates/two_qpo/results/{candidate}_result.json")
        # log_bfs_one_qpo.append(res_one_qpo.log_evidence - res_no_qpo.log_evidence)
        log_bfs_one_qpo.append(res_one_qpo.log_bayes_factor)
        # log_bfs_two_qpo.append(res_two_qpo.log_evidence - res_no_qpo.log_evidence)
        log_P_samples = np.array(res_one_qpo.posterior['kernel:terms[1]:log_P'])
        frequency_samples = 1 / np.exp(log_P_samples)
        # if log_bfs_one_qpo[-1] > 4:
        #     plt.hist(frequency_samples, bins="fd", density=True)
        #     plt.xlabel('frequency [Hz]')
        #     plt.ylabel('normalised PDF')
        #     median = np.median(frequency_samples)
        #     percentiles = np.percentile(frequency_samples, [16, 84])
        #     plt.title(
        #         f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}, ln BF = {log_bfs_one_qpo[-1]:.2f}")
        #     plt.savefig(f"sliding_window_{band}_candidates/frequency_posterior_{candidate}")
        #     plt.clf()
    except Exception as e:
        print(e)
        log_bfs_one_qpo.append(np.nan)
        log_bfs_two_qpo.append(np.nan)
    print(f"{candidate} one qpo: {log_bfs_one_qpo[-1]}")
    # print(f"{candidate} two qpo: {log_bfs_two_qpo[-1]}")
np.savetxt(f'sliding_window_{band}_candidates/log_bfs_one_qpo', np.array(log_bfs_one_qpo))
# np.savetxt(f'sliding_window_{band}_candidates/log_bfs_two_qpo', np.array(log_bfs_two_qpo))


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
