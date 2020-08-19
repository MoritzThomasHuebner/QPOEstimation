import bilby
import matplotlib.pyplot as plt

segments = []
mean_log_bfs = []

for i in range(31):
    segments.append(i)
    log_bfs = []
    mean_log_bf = 0
    mean_log_bfs.append(mean_log_bf)
    for j in range(9):
        try:
            res_qpo = bilby.result.read_in_result(f"sliding_window/period_{j}/{i}_qpo_result.json")
            res_no_qpo = bilby.result.read_in_result(f"sliding_window/period_{j}/{i}_no_qpo_result.json")
            log_bf = res_qpo.log_evidence - res_no_qpo.log_evidence
            mean_log_bf += log_bf
            print(f"{j} {i}: {log_bf}")
        except Exception:
            pass
    print(f"{i}: {mean_log_bf}")

plt.plot(segments, mean_log_bfs)
plt.savefig("mean_log_bfs")
plt.clf()