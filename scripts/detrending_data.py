import os
from pathlib import Path

import bilby
import celerite
import matplotlib
import matplotlib.pyplot as plt

import QPOEstimation
from QPOEstimation.likelihood import CeleriteLikelihood, QPOTerm, PoissonLikelihoodWithBackground, GrothLikelihood
from QPOEstimation.model.series import *
from QPOEstimation.stabilisation import bar_lev

matplotlib.use('Qt5Agg')
run_id = 0
period_number = 0
n_qpos = 1
model_id = 0


likelihood_model = 'gaussian_process'
band = '16_32Hz'
band_maximum = 64
sampling_frequency = 4*band_maximum

data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz.dat')

times = data[:, 0]
counts = data[:, 1]


pulse_period = 7.56  # see papers
start = 20.0

segment_length = 500.0

for i in range(36):
    stop = start + segment_length

    indices = np.where(np.logical_and(times > start, times < stop))
    t = times[indices]
    c = counts[indices]
    c = c.astype(int)

    if n_qpos == 0:
        result = bilby.result.read_in_result(f"sliding_window_{band}/period_{period_number}/no_qpo/results/0_result.json")
    elif n_qpos == 1:
        result = bilby.result.read_in_result(f"sliding_window_{band}/period_{period_number}/one_qpo/results/0_result.json")
    else:
        result = bilby.result.read_in_result(f"sliding_window_{band}/period_{period_number}/two_qpo/results/0_result.json")


    priors = bilby.core.prior.PriorDict()

    stabilised_counts = bar_lev(c)
    stabilised_variance = np.ones(len(stabilised_counts))

    # A non-periodic component
    Q = 1.0 / np.sqrt(2.0)
    w0 = 3.0
    S0 = np.var(stabilised_counts) / (w0 * Q)

    if n_qpos == 0 or n_qpos == 1:
        kernel = QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)
    elif n_qpos == 2:
        kernel = QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3) \
                 + QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)

    params_dict = kernel.get_parameter_dict()
    print(params_dict)
    gp = celerite.GP(kernel, mean=np.mean(stabilised_counts))
    gp.compute(t, stabilised_variance)  # You always need to call compute once.

    if n_qpos == 0:
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='log_a')
        priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=10, name='log_b')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(sampling_frequency), name='log_c')
        priors['kernel:log_P'] = bilby.core.prior.DeltaFunction(peak=-2, name='log_P')
    elif n_qpos == 1:
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='log_a')
        priors['kernel:log_b'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='log_b')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(sampling_frequency), name='log_c')
        priors['kernel:log_P'] = bilby.core.prior.Uniform(minimum=np.log(7.56/2), maximum=np.log(7.56*8), name='log_P')

    elif n_qpos == 2:
        priors = bilby.core.prior.ConditionalPriorDict()
        priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[0]:log_a')
        priors['kernel:terms[0]:log_b'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='terms[0]:log_b')
        priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(sampling_frequency), name='terms[0]:log_c')
        priors['kernel:terms[0]:log_P'] = bilby.core.prior.Uniform(minimum=np.log(7.56*0.75), maximum=np.log(7.56*8), name='terms[0]:log_P')
        priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[1]:log_a')
        priors['kernel:terms[1]:log_b'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='terms[1]:log_b')
        priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[1]:log_c')
        priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=np.log(7.56*0.25), maximum=np.log(7.56*0.75), name='terms[1]:log_P')



    max_like_params = result.posterior.iloc[-1]
    for name, value in max_like_params.items():
        try:
            gp.set_parameter(name=name, value=value)
        except ValueError:
            continue


    idxs = np.where(np.logical_and(20 + i*10 <= t, t <= 20 + 30 + i*10))[0]
    x = t[idxs]
    pred_mean, pred_var = gp.predict(stabilised_counts, x, return_var=True)
    pred_std = np.sqrt(pred_var)
    plt.legend()

    stabilised_counts = stabilised_counts[idxs]
    # stabilised_variance = stabilised_variance[idxs]
    # color = "#ff7f0e"
    # plt.errorbar(x, stabilised_counts, yerr=stabilised_variance, fmt=".k", capsize=0, label='data')
    # plt.plot(x, pred_mean, color=color, label='Prediction')
    # plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
    #                  edgecolor="none")
    # plt.xlabel("time [s]")
    # plt.ylabel("variance stabilised data")
    # plt.show()
    # plt.clf()
    #
    # plt.errorbar(x, stabilised_counts - pred_mean, yerr=stabilised_variance, fmt=".k", capsize=0, label='data')
    # plt.xlabel("time [s]")
    # plt.ylabel("variance stabilised data")
    # plt.show()
    # plt.clf()
    #
    # idxs = np.where(np.logical_and(20 < t, t < 50))[0]
    truncated_times = x[int(len(x)/3):int(len(x)*2/3)]
    truncated_detrend_counts = stabilised_counts[int(len(x)/3):int(len(x)*2/3)] - pred_mean[int(len(x)/3):int(len(x)*2/3)]
    np.savetxt(f'detrend_counts_{i}.dat', np.array([truncated_times, truncated_detrend_counts]).T)

    # plt.hist(truncated_detrend_counts, bins='fd')
    # plt.show()
    # print(np.mean(truncated_detrend_counts))
    # print(np.std(truncated_detrend_counts))