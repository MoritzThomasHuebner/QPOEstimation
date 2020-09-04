import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use("Qt5Agg")
from copy import deepcopy
candidates = np.arange(0, 89)

n_candidates = 89

for candidate in range(len(candidates)):
    log_bfs_one_qpo = []
    try:
        res_no_qpo = bilby.result.read_in_result(f"sliding_window_candidates/no_qpo/{candidate}_result.json")
        res_one_qpo = bilby.result.read_in_result(f"sliding_window_candidates/one_qpo/{candidate}_result.json")
        log_bfs_one_qpo.append(res_one_qpo.log_evidence - res_no_qpo.log_evidence)
        log_P_samples = np.array(res_one_qpo.posterior['kernel:terms[1]:log_P'])
        frequency_samples = 1 / np.exp(log_P_samples)
        plt.hist(frequency_samples, bins="fd", density=True)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('normalised PDF')
        median = np.median(frequency_samples)
        percentiles = np.percentile(frequency_samples, [16, 84])
        plt.title(
            f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
        plt.savefig(f"sliding_window_candidates/frequency_posterior_{candidate}")
        plt.clf()
    except Exception as e:
        print(e)
        log_bfs_one_qpo.append(np.nan)
    print(f"{candidate} one qpo: {log_bfs_one_qpo[-1]}")
np.savetxt(f'sliding_window_candidates/log_bfs_one_qpo_{candidate}', np.array(log_bfs_one_qpo))


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
