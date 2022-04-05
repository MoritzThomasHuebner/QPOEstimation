import celerite
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import QPOEstimation.likelihood
from QPOEstimation.result import GPResult
from QPOEstimation.utils import get_injection_outdir, get_injection_label

# matplotlib.rcParams['text.usetex'] = True
plt.style.use("paper.mplstyle")

base_injection_outdir="results/injection_files_mss_with_mean"

samples = []
outdir_qpo_qpo = get_injection_outdir(
    injection_mode="qpo_plus_red_noise", recovery_mode="qpo_plus_red_noise",
    likelihood_model="celerite", base_injection_outdir=base_injection_outdir)
outdir_qpo_qpo = f"{outdir_qpo_qpo}/results"
outdir_qpo_red_noise = get_injection_outdir(
    injection_mode="qpo_plus_red_noise", recovery_mode="red_noise",
    likelihood_model="celerite", base_injection_outdir=base_injection_outdir)
outdir_qpo_red_noise = f"{outdir_qpo_red_noise}/results"
outdir_red_noise_red_noise = get_injection_outdir(
    injection_mode="red_noise", recovery_mode="red_noise",
    likelihood_model="celerite", base_injection_outdir=base_injection_outdir)
outdir_red_noise_red_noise = f"{outdir_red_noise_red_noise}/results"
outdir_red_noise_qpo = get_injection_outdir(
    injection_mode="red_noise", recovery_mode="qpo_plus_red_noise",
    likelihood_model="celerite", base_injection_outdir=base_injection_outdir)
outdir_red_noise_qpo = f"{outdir_red_noise_qpo}/results"

ln_bfs_qpo_inj = []
ln_bfs_red_noise_inj = []
ln_bfs_high_amp_qpo_inj = []

# for injection_id in range(0, 1000):
#     print(injection_id)
#     label = get_injection_label(run_mode="entire_segment", injection_id=injection_id) + "_1_skew_gaussians"
#     label_high_qpo_amp = get_injection_label(
#         run_mode="entire_segment", injection_id=injection_id + 1000) + "_1_skew_gaussians"
#     try:
#         # res_qpo_qpo = GPResult.from_json(outdir=outdir_qpo_qpo, label=label)
#         # res_qpo_red_noise = GPResult.from_json(outdir=outdir_qpo_red_noise, label=label)
#         res_high_amp_qpo_qpo = GPResult.from_json(outdir=outdir_qpo_qpo, label=label_high_qpo_amp)
#         res_high_amp_qpo_red_noise = GPResult.from_json(outdir=outdir_qpo_red_noise, label=label_high_qpo_amp)
#         # res_red_noise_red_noise = GPResult.from_json(outdir=outdir_red_noise_red_noise, label=label)
#         # res_red_noise_qpo = GPResult.from_json(outdir=outdir_red_noise_qpo, label=label)
#         # ln_bfs_qpo_inj.append(res_qpo_qpo.log_evidence - res_qpo_red_noise.log_evidence)
#         ln_bfs_high_amp_qpo_inj.append(res_high_amp_qpo_qpo.log_evidence - res_high_amp_qpo_red_noise.log_evidence)
#         # ln_bfs_red_noise_inj.append(res_red_noise_qpo.log_evidence - res_red_noise_red_noise.log_evidence)
#         print(ln_bfs_high_amp_qpo_inj[-1])
#     except (OSError, FileNotFoundError) as e:
#         print(e)
#         continue

# np.savetxt("results/ln_bfs_qpo_inj_mss.txt", ln_bfs_qpo_inj)
# np.savetxt("results/ln_bfs_high_amp_qpo_inj_mss.txt", ln_bfs_high_amp_qpo_inj)
# np.savetxt("results/ln_bfs_red_noise_inj_mss.txt", ln_bfs_red_noise_inj)

ln_bfs_qpo_inj = np.loadtxt("results/ln_bfs_qpo_inj_mss.txt")
ln_bfs_high_amp_qpo_inj = np.loadtxt("results/ln_bfs_high_amp_qpo_inj_mss.txt")
ln_bfs_red_noise_inj = np.loadtxt("results/ln_bfs_red_noise_inj_mss.txt")

least_sig_qpo = np.argmin(ln_bfs_high_amp_qpo_inj) + 1000
most_sig_qpo = np.argmax(ln_bfs_high_amp_qpo_inj) + 1000

bins = np.arange(-5, 70)
# bins = "fd"
plt.hist(ln_bfs_qpo_inj, alpha=0.3, density=True, bins=bins, label="Red noise + QPO")
plt.hist(ln_bfs_high_amp_qpo_inj, alpha=0.3, density=True, bins=bins, label="Red noise + high amp. QPO")
plt.hist(ln_bfs_red_noise_inj, alpha=0.3, density=True, bins=bins, label="Red noise")
plt.semilogy()
plt.xlabel("$\ln BF_{\mathrm{QPO}}$")
plt.ylabel("$p(\ln BF_{\mathrm{QPO}})$")
plt.legend()
plt.tight_layout()
plt.savefig("results/mss_plot_with_mean.pdf")
plt.show()

print(len(np.where(np.array(ln_bfs_qpo_inj) > 0)[0]))
print(len(np.where(np.array(ln_bfs_high_amp_qpo_inj) > 0)[0]))
print(len(np.where(np.array(ln_bfs_red_noise_inj) > 0)[0]))

least_sig_qpo_data = np.loadtxt(f"injections/injection_files_mss_with_mean/qpo_plus_red_noise/celerite/{least_sig_qpo}_data.txt")
most_sig_qpo_data = np.loadtxt(f"injections/injection_files_mss_with_mean/qpo_plus_red_noise/celerite/{most_sig_qpo}_data.txt")

import json
with open("injections/injection_files_mss_with_mean/qpo_plus_red_noise/celerite/1001_params.json", "r") as f:
    params = json.load(f)

kernel = QPOEstimation.likelihood.get_kernel("qpo_plus_red_noise")
mean = QPOEstimation.likelihood.get_mean_model("skew_gaussian", n_components=1)

figure, axes = plt.subplots(ncols=1, nrows=2, sharex='col', sharey='none', figsize=(8, 6))
axes = axes.ravel()

for data, ax in zip([least_sig_qpo_data, most_sig_qpo_data], axes):
    times = data[:, 0]
    y = data[:, 1]
    yerr = data[:, 2]
    likelihood = QPOEstimation.likelihood.CeleriteLikelihood(
        mean_model=mean, kernel=kernel, t=times, y=y, yerr=yerr)
    likelihood.parameters = params

    color = "#ff7f0e"
    ax.errorbar(times, y, yerr=yerr, fmt=".k", capsize=0, label="data")
    x = np.linspace(times[0], times[-1], 5000)
    gp = celerite.GP(kernel=kernel, mean=mean)
    for k, v in params.items():
        gp.set_parameter(k, v)

    gp.compute(times, yerr=yerr)
    pred_mean, pred_var = gp.predict(y, x, return_var=True)
    pred_std = np.sqrt(pred_var)
    ax.plot(x, pred_mean, color=color, label="Prediction")
    ax.fill_between(x, pred_mean + pred_std, pred_mean - pred_std,
                    color=color, alpha=0.3, edgecolor="none")

    ax.plot(times, gp.mean.get_value(times), color="green", label="Mean function")
    ax.set_ylabel("y")

axes[1].set_xlabel("time [s]")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
figure.legend(by_label.values(), by_label.keys(), ncol=3, loc="upper center")
plt.subplots_adjust(hspace=0.04)
plt.savefig(f"results/injections_mss_extreme_cases.pdf", dpi=150)
plt.show()
plt.clf()
