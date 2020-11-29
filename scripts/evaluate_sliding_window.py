import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from QPOEstimation.stabilisation import bar_lev
# matplotlib.use("Qt5Agg")
from copy import deepcopy

segments = np.arange(0, 31)
# segments = np.arange(0, 8)
mean_log_bfs = []

import numpy as np
n_periods = 47
period_one_log_bf_data = []
period_two_log_bf_data = []

band_minimum = 5
band_maximum = 64

pulse_period = 7.56
segment_step = 0.23625
segment_length = 0.945
data_mode = "normal"
likelihood_model = "gaussian_process"
alpha = 0.02

if band_maximum <= 64:
    sampling_frequency = 256
elif band_maximum <= 128:
    sampling_frequency = 512
else:
    sampling_frequency = 1024

band = f'{band_minimum}_{band_maximum}Hz'
suffix = ''

outdir = f'sliding_window_{band}_{data_mode}'

if data_mode == 'smoothed':
    data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_exp_smoothed_alpha_{alpha}.dat')
elif data_mode == 'smoothed_residual':
    data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_exp_residual_alpha_{alpha}.dat')
else:
    data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz.dat')

times = data[:, 0]
counts = data[:, 1]



for period in range(n_periods):
    log_bfs_red_noise = []
    log_bfs_qpo = []
    log_bfs_mixed = []
    mean_frequency_qpo = []
    std_frequency_qpo = []
    mean_frequency_mixed = []
    std_frequency_mixed = []

    for run_id in range(len(segments)):
        try:
            res_qpo = bilby.result.read_in_result(f"{outdir}/period_{period}/qpo/results/{run_id}_{likelihood_model}_result.json")
            res_red_noise = bilby.result.read_in_result(f"{outdir}/period_{period}/red_noise/results/{run_id}_{likelihood_model}_result.json")
            res_mixed = bilby.result.read_in_result(f"{outdir}/period_{period}/mixed/results/{run_id}_result.json")
            log_bf_qpo = res_qpo.log_bayes_factor
            log_bf_red_noise = res_red_noise.log_bayes_factor
            log_bf_mixed = res_mixed.log_bayes_factor

            log_f_samples_qpo = np.array(res_qpo.posterior['kernel:log_f'])
            frequency_samples_qpo = np.exp(log_f_samples_qpo)
            mean_frequency_qpo.append(np.mean(frequency_samples_qpo))
            std_frequency_qpo.append(np.std(frequency_samples_qpo))

            log_f_samples_mixed = np.array(res_mixed.posterior['kernel:terms[0]:log_f'])
            frequency_samples_mixed = np.exp(log_f_samples_mixed)
            mean_frequency_qpo.append(np.mean(frequency_samples_mixed))
            std_frequency_qpo.append(np.std(frequency_samples_mixed))

            # res_no_qpo_whittle = bilby.result.read_in_result(f"{outdir}/period_{period}/no_qpo/results/{run_id}_whittle_result.json")
            # res_one_qpo_whittle = bilby.result.read_in_result(f"{outdir}/period_{period}/one_qpo/results/{run_id}_whittle_result.json")
            # log_bf_one_qpo_whittle = res_one_qpo_whittle.log_evidence - res_no_qpo_whittle.log_evidence
            # frequency_samples_whittle = np.array(res_one_qpo_whittle.posterior['central_frequency'])
            # mean_frequency_whittle.append(np.mean(frequency_samples_whittle))
            # std_frequency_whittle.append(np.std(frequency_samples_whittle))
        except Exception as e:
            print(e)
            log_bf_qpo = np.nan
            log_bf_red_noise = np.nan
            mean_frequency_qpo.append(np.nan)
            std_frequency_qpo.append(np.nan)
        log_bfs_qpo.append(log_bf_qpo)
        log_bfs_red_noise.append(log_bf_red_noise)
        log_bfs_mixed.append(log_bf_mixed)

        print(f"{period} {run_id} one qpo: {log_bf_qpo}")
        print(f"{period} {run_id} red noise: {log_bf_red_noise}")
        print(f"{period} {run_id} mixed: {log_bf_mixed}")

    np.savetxt(f'{outdir}/log_bfs_period_qpo_{period}', np.array(log_bfs_qpo))
    np.savetxt(f'{outdir}/log_bfs_period_red_noise_{period}', np.array(log_bfs_red_noise))
    np.savetxt(f'{outdir}/log_bfs_period_mixed_{period}', np.array(log_bfs_red_noise))
    np.savetxt(f'{outdir}/mean_frequencies_{period}', np.array(mean_frequency_qpo))
    np.savetxt(f'{outdir}/std_frequencies_{period}', np.array(std_frequency_qpo))

    # start_times = period * pulse_period + segments * segment_step + 20
    xs = np.arange(len(segments))
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('segment start time [s]')
    ax1.set_ylabel('ln BF', color=color)
    ax1.plot(xs, log_bfs_qpo, color=color, ls='solid', label='One QPO vs white noise')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('frequency [Hz]', color=color)  # we already handled the x-label with ax1
    ax2.plot(xs, mean_frequency_qpo, color=color)
    mean_frequency_qpo = np.array(mean_frequency_qpo)
    std_frequency_qpo = np.array(std_frequency_qpo)
    plt.fill_between(xs, mean_frequency_qpo + std_frequency_qpo, mean_frequency_qpo - std_frequency_qpo, color=color, alpha=0.3,
                     edgecolor="none")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend()
    plt.savefig(f'{outdir}/log_bfs_period_{period}')
    plt.clf()


    # start_times = period * pulse_period + segments * segment_step + 20
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('segment start time [s]')
    ax1.set_ylabel('ln BF', color=color)
    ax1.plot(xs, np.array(log_bfs_qpo) - np.array(log_bfs_red_noise), color=color, ls='solid', label='One QPO vs Red noise')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    mean_frequency_qpo = np.array(mean_frequency_qpo)
    std_frequency_qpo = np.array(std_frequency_qpo)
    color = 'tab:blue'
    ax2.set_ylabel('frequency [Hz]', color=color)  # we already handled the x-label with ax1
    ax2.plot(xs, mean_frequency_qpo, color=color)
    plt.fill_between(xs, mean_frequency_qpo + std_frequency_qpo, mean_frequency_qpo - std_frequency_qpo, color=color, alpha=0.3,
                     edgecolor="none")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend()
    plt.savefig(f'{outdir}/log_bfs_period_{period}_qpo_vs_red_noise')
    plt.clf()

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('segment start time [s]')
    ax1.set_ylabel('ln BF', color=color)
    ax1.plot(xs, np.array(log_bfs_mixed) - np.array(log_bfs_red_noise), color=color, ls='solid', label='QPO and Red noise vs Red noise')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    mean_frequency_mixed = np.array(mean_frequency_mixed)
    std_frequency_mixed = np.array(std_frequency_mixed)
    color = 'tab:blue'
    ax2.set_ylabel('frequency [Hz]', color=color)  # we already handled the x-label with ax1
    ax2.plot(xs, mean_frequency_mixed, color=color)
    plt.fill_between(xs, mean_frequency_mixed + std_frequency_mixed, mean_frequency_mixed - std_frequency_mixed, color=color, alpha=0.3,
                     edgecolor="none")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend()
    plt.savefig(f'{outdir}/log_bfs_period_{period}_mixed_vs_red_noise')
    plt.clf()


    # fig, ax1 = plt.subplots()
    # color = 'tab:red'
    # ax1.set_xlabel('segment start time [s]')
    # ax1.set_ylabel('ln BF', color=color)
    # ax1.plot(start_times, log_bfs_one_qpo_whittle, color=color, ls='solid', label='One QPO')
    # # ax1.plot(start_times, log_bfs_two_qpo, color=color, ls='dotted', label='Two QPOs')
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    # color = 'tab:blue'
    # ax2.set_ylabel('frequency [Hz]', color=color)  # we already handled the x-label with ax1
    # ax2.plot(start_times, mean_frequency_whittle, color=color)
    # mean_frequency_whittle = np.array(mean_frequency_whittle)
    # std_frequency_whittle = np.array(std_frequency_whittle)
    # plt.fill_between(start_times, mean_frequency_whittle + std_frequency_whittle,
    #                  mean_frequency_whittle - std_frequency_whittle,
    #                  color=color, alpha=0.3, edgecolor="none")
    # ax2.tick_params(axis='y', labelcolor=color)
    #
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # ax1.legend()
    # plt.savefig(f'{outdir}/log_bfs_period_{period}_whittle')
    # plt.clf()
    # period_one_log_bf_data.append(deepcopy(log_bfs_one_qpo))
    # period_two_log_bf_data.append(deepcopy(log_bfs_two_qpo))


