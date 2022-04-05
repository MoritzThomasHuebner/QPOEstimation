import bilby.core.result
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("paper.mplstyle")

xs = np.arange(1, 21)
times = np.arange(10, 210, 10.)
print(times)


ln_bfs_gp_stat = []
ln_bfs_gp_non_stat = []
ln_bfs_whittle = []
ln_bfs_gp_non_stat_vs_stat = []

for t in times:
    label = f"01_-{t}_{t}_1_0s"
    label_whittle = f"01_-{t}_{t}"
    outdir_base = "results/injection_non_stat_bias/qpo_plus_red_noise_injection"
    res_gp_stat_qpo_plus_red_noise = bilby.result.read_in_result(outdir=f"{outdir_base}/qpo_plus_red_noise_recovery/celerite/results/", label=label)
    res_gp_stat_red_noise = bilby.result.read_in_result(outdir=f"{outdir_base}/red_noise_recovery/celerite/results/", label=label)
    res_gp_non_stat_qpo_plus_red_noise = bilby.result.read_in_result(outdir=f"{outdir_base}/qpo_plus_red_noise_recovery/celerite_windowed/results/", label=label)
    res_gp_non_stat_red_noise = bilby.result.read_in_result(outdir=f"{outdir_base}/red_noise_recovery/celerite_windowed/results/", label=label)
    res_whittle_qpo_plus_red_noise = bilby.result.read_in_result(outdir=f"{outdir_base}/qpo_plus_red_noise_recovery/whittle/results/", label=label_whittle)
    res_whittle_red_noise = bilby.result.read_in_result(outdir=f"{outdir_base}/red_noise_recovery/whittle/results/", label=label_whittle)

    ln_bfs_gp_stat.append(res_gp_stat_qpo_plus_red_noise.log_evidence - res_gp_stat_red_noise.log_evidence)
    ln_bfs_gp_non_stat.append(res_gp_non_stat_qpo_plus_red_noise.log_evidence - res_gp_non_stat_red_noise.log_evidence)
    ln_bfs_whittle.append(res_whittle_qpo_plus_red_noise.log_evidence - res_whittle_red_noise.log_evidence)
    ln_bfs_gp_non_stat_vs_stat.append(res_gp_non_stat_qpo_plus_red_noise.log_evidence - res_gp_stat_qpo_plus_red_noise.log_evidence)

plt.plot(xs, ln_bfs_whittle, label="Periodogram")
plt.plot(xs, ln_bfs_gp_stat, label="Stat. GP")
plt.plot(xs, ln_bfs_gp_non_stat, label="Non stat. GP")
plt.xlim(1., 20)
plt.ylim(-1, 50)
plt.xlabel(r"$x$")
plt.xticks([1, 5, 10, 15, 20], [1, 5, 10, 15, 20])
plt.ylabel(r"$\ln BF_{\mathrm{QPO}}$")
plt.legend()
plt.tight_layout()
plt.savefig("results/non_stat_comparison.pdf")
plt.show()

plt.plot(xs, ln_bfs_gp_non_stat_vs_stat)
plt.xlim(1., 20)
plt.ylim(-1, 175)
plt.xticks([1, 5, 10, 15, 20], [1, 5, 10, 15, 20])
plt.xlabel(r"$x$")
plt.ylabel(r"$\ln BF_{\mathrm{non-stat.}}$")
plt.tight_layout()
plt.savefig("results/non_stat_vs_stat_gp.pdf")
plt.show()