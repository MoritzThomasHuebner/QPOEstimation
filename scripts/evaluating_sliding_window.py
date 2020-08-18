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
            res = bilby.result.read_in_result(f"sliding_window/period_{j}/{i}_result.json")
            mean_log_bf += res.log_bayes_factor / 9
            print(f"{j} {i}: {res.log_bayes_factor}")
        except Exception:
            pass
    print(f"{i}: {mean_log_bf}")

plt.plot(segments, mean_log_bfs)
plt.savefig("mean_log_bfs")
plt.clf()