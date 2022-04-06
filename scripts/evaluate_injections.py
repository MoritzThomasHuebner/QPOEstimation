import argparse
import sys

import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from QPOEstimation.utils import get_injection_outdir
from QPOEstimation.parse import MODES

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--injection_mode", default="qpo", choices=MODES, type=str)
    parser.add_argument("--n_injections", default=100, type=int)
    args = parser.parse_args()

    injection_mode = args.injection_mode
    n_injections = args.n_injections
else:
    matplotlib.use("Qt5Agg")
    injection_mode = "red_noise"
    n_injections = 100

likelihood_model = "celerite"
injections = np.arange(0, n_injections)


log_bfs_one_qpo_gpr = []


# averaged runs

minimum_log_a = -2
maximum_log_a = 1
minimum_log_c = 1
maximum_log_c = 5
minimum_log_f = np.log(10)
maximum_log_f = np.log(64)

log_as = np.linspace(minimum_log_a, maximum_log_a, 10)
log_cs = np.linspace(minimum_log_c, maximum_log_c, 10)
if injection_mode == "qpo":
    log_fs = np.linspace(minimum_log_f, maximum_log_f, 10)
else:
    log_fs = [0]

log_evidences_qpo = []
log_evidences_red_noise = []
averaged_log_bfs_qpo_v_red_noise = []
averaged_log_bfs_qpo_v_red_noise_err = []
for injection_id in range(100):
    log_a = log_as[int(str(injection_id).zfill(3)[1])]
    log_c = log_cs[int(str(injection_id).zfill(3)[2])]
    log_f = np.log(20)
    individual_log_bfs_qpo_v_red_noise = []
    for j in range(10):
        try:
            res_qpo = bilby.result.read_in_result(
                outdir=get_injection_outdir(injection_mode=injection_mode, recovery_mode="qpo",
                                            likelihood_model=likelihood_model), label=f"{str(injection_id).zfill(2)}")
            res_red_noise = bilby.result.read_in_result(
                outdir=get_injection_outdir(injection_mode=injection_mode, recovery_mode="red_noise",
                                            likelihood_model=likelihood_model), label=f"{str(injection_id).zfill(2)}")
            individual_log_bfs_qpo_v_red_noise.append((res_qpo.log_evidence - res_red_noise.log_evidence))
        except Exception as e:
            print(e)
    averaged_log_bfs_qpo_v_red_noise.append(np.mean(individual_log_bfs_qpo_v_red_noise))
    averaged_log_bfs_qpo_v_red_noise_err.append(np.std(individual_log_bfs_qpo_v_red_noise))

    print(averaged_log_bfs_qpo_v_red_noise[-1])
    print(injection_id)

for i in range(10):
    plt.errorbar(log_as, averaged_log_bfs_qpo_v_red_noise[i::10],
                 yerr=averaged_log_bfs_qpo_v_red_noise_err[i::10], label=f"ln c = {log_cs[i]:.2f}")
    plt.xlabel("ln a")
    plt.ylabel("ln BF")
    plt.legend()
    suffix = "20Hz"
    plt.title("10 runs averaged")
    if injection_mode == "qpo":
        plt.savefig(f"results/ln_a_v_ln_BF_{injection_mode}_{suffix}_{i}.png")
    else:
        plt.savefig(f"results/ln_a_v_ln_BF_{injection_mode}_{i}.png")
    plt.show()
    plt.clf()


for i in range(10):
    plt.semilogy(log_as, averaged_log_bfs_qpo_v_red_noise_err[i::10], label=f"ln c = {log_cs[i]:.2f}")
    plt.xlabel("ln a")
    plt.ylabel("$\Delta \ln \mathrm{BF}$")
plt.legend()
suffix = "20Hz"
plt.title("Standard deviation based on 10 injections")
if injection_mode == "qpo":
    plt.savefig(f"results/ln_a_v_ln_BF_{injection_mode}_{suffix}_errs.png")
else:
    plt.savefig(f"results/ln_a_v_ln_BF_{injection_mode}_errs.png")
plt.show()
plt.clf()

for i in range(10):
    plt.errorbar(log_cs, averaged_log_bfs_qpo_v_red_noise[10 * i: 10 * i + 10],
                 yerr=averaged_log_bfs_qpo_v_red_noise_err[10 * i: 10 * i + 10], label=f"ln a = {log_as[i]:.2f}")
    plt.xlabel("ln c")
    plt.ylabel("ln BF")
    plt.legend()
    plt.title("10 runs averaged")
    if injection_mode == "qpo":
        plt.savefig(f"results/ln_c_v_ln_BF_{injection_mode}_{suffix}_{i}.png")
    else:
        plt.savefig(f"results/ln_c_v_ln_BF_{injection_mode}_{i}.png")
    plt.show()
    plt.clf()

for i in range(10):
    plt.semilogy(log_cs, averaged_log_bfs_qpo_v_red_noise_err[10 * i: 10 * i + 10], label=f"ln a = {log_as[i]:.2f}")
    plt.xlabel("ln c")
    plt.ylabel("$\Delta \ln \mathrm{BF}$")
plt.legend()
plt.title("Standard deviation based on 10 injections")
if injection_mode == "qpo":
    plt.savefig(f"results/ln_c_v_ln_BF_{injection_mode}_{suffix}_errs.png")
else:
    plt.savefig(f"results/ln_c_v_ln_BF_{injection_mode}_errs.png")

plt.show()
plt.clf()


log_bfs_qpo_red_noise_reshaped = np.reshape(averaged_log_bfs_qpo_v_red_noise, (10, 10))

cmap = matplotlib.cm.jet
ax = plt.contourf(log_as, log_cs, log_bfs_qpo_red_noise_reshaped,
                  cmap=cmap, levels=np.linspace(np.amin(log_bfs_qpo_red_noise_reshaped),
                                                np.amax(log_bfs_qpo_red_noise_reshaped), 1000))
plt.colorbar(ax)
plt.xlabel("ln c")
plt.ylabel("ln a")
plt.title("10 runs averaged")
if injection_mode == "qpo":
    plt.savefig(f"results/ln_a_v_ln_c_v_ln_BF_{injection_mode}_{suffix}.png")
else:
    plt.savefig(f"results/ln_a_v_ln_c_v_ln_BF_{injection_mode}")
plt.show()
plt.clf()
