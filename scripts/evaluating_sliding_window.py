import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use("Qt5Agg")
from copy import deepcopy
segments = np.arange(0, 31)
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

period_one_log_bf_data = []
period_two_log_bf_data = []

for j in range(9):
    log_bfs_one_qpo = []
    log_bfs_two_qpo = []
    for i in range(31):
        try:
            res_no_qpo = bilby.result.read_in_result(f"sliding_window/period_{j}/no_qpo/{i}_no_qpo_result.json")
            res_qpo = bilby.result.read_in_result(f"sliding_window/period_{j}/one_qpo/{i}_one_qpo_result.json")
            res_two_qpo = bilby.result.read_in_result(f"sliding_window/period_{j}/two_qpo/{i}_two_qpo_result.json")
            log_bf_one_qpo = res_qpo.log_evidence - res_no_qpo.log_evidence
            log_bf_two_qpo = res_two_qpo.log_evidence - res_no_qpo.log_evidence
        except Exception:
            log_bf_one_qpo = np.nan
            log_bf_two_qpo = np.nan
        log_bfs_one_qpo.append(log_bf_one_qpo)
        log_bfs_two_qpo.append(log_bf_two_qpo)
        print(f"{j} {i}: {log_bf_one_qpo}")
    period_one_log_bf_data.append(deepcopy(log_bfs_one_qpo))
    period_two_log_bf_data.append(deepcopy(log_bfs_two_qpo))

for j in range(9):
    plt.plot(segments, period_one_log_bf_data[j], label=f'period_{j}')
    plt.legend()
    plt.xlabel("Data segment")
    plt.ylabel("ln BF")
    plt.savefig("one_qpo_log_bfs")
    plt.clf()

for j in range(9):
    plt.plot(segments, period_two_log_bf_data[j], label=f'period_{j}')
    plt.legend()
    plt.xlabel("Data segment")
    plt.ylabel("ln BF")
    plt.savefig("two_qpo_log_bfs")
    plt.clf()
