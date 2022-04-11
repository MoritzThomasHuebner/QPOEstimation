import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from QPOEstimation.stabilisation import bar_lev
# matplotlib.use("Qt5Agg")
from copy import deepcopy

segments = np.arange(0, 8)
mean_log_bfs = []

n_periods = 47
period_one_log_bf_data = []
period_two_log_bf_data = []

band_minimum = 5
band_maximum = 64

pulse_period = 7.56
segment_step = 0.945
segment_length = 2.8
kernel_type = "qpo_plus_red_noise"
likelihood_model = "celerite_windowed"
alpha = 0.02

run_mode = "sliding_window"

if band_maximum <= 64:
    sampling_frequency = 256
elif band_maximum <= 128:
    sampling_frequency = 512
else:
    sampling_frequency = 1024

band = f"{band_minimum}_{band_maximum}Hz"
suffix = ""
outdir = f"results/SGR_1806_20/{run_mode}/{band}"
outdir_qpo_model = f"{outdir}/{kernel_type}/{likelihood_model}/"
outdir_red_noise = f"{outdir}/red_noise/{likelihood_model}/"

data = np.loadtxt(f"data/sgr1806_{sampling_frequency}Hz.dat")

times = data[:, 0]
counts = data[:, 1]


for period in range(n_periods):
    log_bfs_qpo_model = []
    mean_frequency_qpo = []
    std_frequency_qpo = []
    for run_id in range(len(segments)):
        try:
            res_qpo_model = bilby.result.read_in_result(f"{outdir_qpo_model}/period_{period}/results/{run_id}_result.json")
            res_red_noise = bilby.result.read_in_result(f"{outdir_red_noise}/period_{period}/results/{run_id}_result.json")
            log_bf_qpo_model = res_qpo_model.log_bayes_factor - res_red_noise.log_bayes_factor

            try:
                log_f_samples_qpo = np.array(res_qpo_model.posterior["kernel:terms[0]:log_f"])
            except Exception as e:
                log_f_samples_qpo = np.array(res_qpo_model.posterior["kernel:log_f"])
            frequency_samples_qpo = np.exp(log_f_samples_qpo)
            mean_frequency_qpo.append(np.mean(frequency_samples_qpo))
            std_frequency_qpo.append(np.std(frequency_samples_qpo))

        except Exception as e:
            print(e)
            log_bf_qpo_model = np.nan
            mean_frequency_qpo.append(np.nan)
            std_frequency_qpo.append(np.nan)
        log_bfs_qpo_model.append(log_bf_qpo_model)

        print(f"{period} {run_id}: {log_bf_qpo_model}")

    np.savetxt(f"{outdir_qpo_model}/log_bfs_{kernel_type}_period_{period}", np.array(log_bfs_qpo_model))
    np.savetxt(f"{outdir_qpo_model}/mean_frequencies_{period}", np.array(mean_frequency_qpo))
    np.savetxt(f"{outdir_qpo_model}/std_frequencies_{period}", np.array(std_frequency_qpo))

    xs = np.arange(len(segments))
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("segment start time [s]")
    ax1.set_ylabel("ln BF", color=color)
    ax1.plot(xs, log_bfs_qpo_model, color=color, ls="solid", label="QPO + Red noise vs Red noise")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("frequency [Hz]", color=color)  # we already handled the x-label with ax1
    ax2.plot(xs, mean_frequency_qpo, color=color)
    mean_frequency_qpo = np.array(mean_frequency_qpo)
    std_frequency_qpo = np.array(std_frequency_qpo)
    plt.fill_between(xs, mean_frequency_qpo + std_frequency_qpo, mean_frequency_qpo - std_frequency_qpo, color=color, alpha=0.3,
                     edgecolor="none")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend()
    plt.savefig(f"{outdir_qpo_model}/log_bfs_period_{period}")
    plt.clf()
