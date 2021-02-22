import argparse
import json
import os
import sys
from pathlib import Path

import bilby
import celerite
import matplotlib
import matplotlib.pyplot as plt
from bilby.core.prior import Prior, Uniform, Beta
from bilby.core.prior.dict import ConditionalPriorDict
from bilby.core.prior.conditional import ConditionalBeta

import QPOEstimation
from QPOEstimation.likelihood import CeleriteLikelihood, WhittleLikelihood, \
    GrothLikelihood, WindowedCeleriteLikelihood, get_kernel
from QPOEstimation.model.celerite import *
from QPOEstimation.model.series import *
from QPOEstimation.prior.gp import *
from QPOEstimation.stabilisation import bar_lev

likelihood_models = ["gaussian_process", "gaussian_process_windowed", "periodogram", "poisson"]
modes = ["qpo", "white_noise", "red_noise", "pure_qpo", "general_qpo"]
run_modes = ['select_time', 'sliding_window', 'multiple_windows', 'candidates', 'injection']
background_models = ["polynomial", "exponential", "mean"]
data_modes = ['normal', 'smoothed', 'smoothed_residual', 'blind_injection']


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'



matplotlib.use('Qt5Agg')

minimum_window_spacing = 0

recovery_mode = "qpo"
likelihood_model = "gaussian_process"
background_model = "fred"


nlive = 300
use_ratio = False

plot = True

suffix = ""


data = np.loadtxt(f'data/SolarFlare/120704187_ctime_lc.txt')
outdir = f"SolarFlare/120704187/"
# outdir = f"SolarFlare/120704187_segments_end/"

# data = np.loadtxt(f'solar_flare_injection/qpo/1_fred_injection_data.txt')
# import json
# with open('solar_flare_injection/qpo/1_fred_injection_params.json') as f:
#     truths = json.load(f)
#     del truths['log_likelihood']
#     del truths['log_prior']
#     try:
#         del truths['kernel:log_b']
#     except Exception:
#         pass
#
# outdir = f"SolarFlare/injection/"




times = data[:, 0]
counts = data[:, 1]
try:
    yerr = data[:, 2]
except Exception:
    yerr = np.sqrt(counts)
# counts = bar_lev(counts)
# yerr = np.ones(len(counts))

# max_rate = np.amax(counts)
# counts /= max_rate

times = times[60:440]
counts = counts[60:440]
yerr = yerr[60:440]

# yerr /= max_rate

def boxcar_filter(cs, n):
    res = np.zeros(len(cs))
    for i, c in enumerate(cs):
        if i < (n - 1)/2:
            boxcar = cs[0: int(i + (n - 1) / 2)]
        elif i > len(cs) - (n - 1)/2:
            boxcar = cs[int(i - (n - 1) / 2):-1]
        else:
            boxcar = cs[int(i - (n - 1) / 2): int(i + (n - 1) / 2)]
        res[i] = sum(boxcar) / len(boxcar)

    return res

# from scipy.signal._savitzky_golay import savgol_filter
# window_length = 151
# filtered_counts = savgol_filter(counts, window_length, 3)
# filtered_counts = boxcar_filter(counts, 21)
# plt.plot(times, counts)
# plt.plot(times, filtered_counts)
# plt.show()

# plt.errorbar(times, counts - filtered_counts, yerr=yerr, fmt=".k", capsize=0, label='data')
# plt.show()

# counts -= filtered_counts
# label = f"savgol_filtered_{recovery_mode}_windowed_{window_length}"


start = times[0]
stop = times[-1]

dt = times[1] - times[0]

sampling_frequency = 1/dt

band_minimum = 1/(stop - start)
band_maximum = sampling_frequency/2
band = f'{band_minimum}_{band_maximum}Hz'

priors = bilby.core.prior.PriorDict()

# kernel = get_kernel(kernel_type=recovery_mode)
# kernel_priors = get_kernel_prior(kernel_type=recovery_mode, min_log_a=-15, max_log_a=10.,
#                                  min_log_c=-15, band_minimum=band_minimum, band_maximum=band_maximum)
# priors.update(kernel_priors)
kernel = celerite.terms.SHOTerm(log_S0=0, log_Q=np.log(1 / np.sqrt(2)), log_omega0=1)
n_shos = 3

for i in range(1, n_shos):
    kernel += celerite.terms.SHOTerm(log_S0=0, log_Q=np.log(1/np.sqrt(2)), log_omega0=1)

# priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name='log_a')
# priors['kernel:terms[0]:log_b'] = bilby.core.prior.DeltaFunction(peak=-20, name='log_b')
# priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=-15, maximum=np.log(band_maximum), name='log_c')
# priors['kernel:terms[0]:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum), maximum=np.log(band_maximum), name='log_f')
# priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name='decay_constraint')

# priors['kernel:terms[1]:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name='log_S0_0')
# priors['kernel:terms[1]:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1/np.sqrt(2)), name='log_a_0')
# priors['kernel:terms[1]:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name='log_omega0_0')
#
# priors['kernel:terms[2]:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name='log_S0_1')
# priors['kernel:terms[2]:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1/np.sqrt(2)), name='log_a_1')
# priors['kernel:terms[2]:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name='log_omega0_1')
#
# priors['kernel:terms[2]:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name='log_S0_2')
# priors['kernel:terms[2]:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1/np.sqrt(2)), name='log_a_2')
# priors['kernel:terms[2]:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name='log_omega0_2')

minimum_spacing = 0
for ii in range(n_shos):
    initial_term = 0
    if ii == 0:
        priors[f'kernel:terms[{ii+initial_term}]:log_S0'] = Beta(minimum=-15, maximum=30, alpha=1, beta=n_shos, name=f'log_S0_{ii}')
    else:
        priors[f'kernel:terms[{ii+initial_term}]:log_S0'] = QPOEstimation.prior.minimum.MinimumPrior(
            order=n_shos-ii, minimum_spacing=minimum_spacing, minimum=-15, maximum=30, name=f'log_S0_{ii}')
    # priors[f'kernel:terms[{ii+initial_term}]:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name=f'log_S0_{ii}')
    # priors[f'kernel:terms[{ii+initial_term}]:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1 / np.sqrt(2)), name=f'log_a_{ii}')
    priors[f'kernel:terms[{ii+initial_term}]:log_Q'] = bilby.core.prior.Uniform(minimum=np.log(1 / np.sqrt(2)), maximum=30, name=f'log_a_{ii}')
    priors[f'kernel:terms[{ii+initial_term}]:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name=f'log_omega0_{ii}')

# label = 'sho_red_noise_kernel'
# mean_model = np.mean(counts)
# kernel = celerite.terms.JitterTerm(log_sigma=-20)
# priors["kernel:log_sigma"] = bilby.core.prior.DeltaFunction(peak=-20, name='log_sigma')


n_models = 1
# label = f'gaussian_{n_models}'


mean_model = get_n_component_mean_model(model=gaussian, n_models=n_models)

mean_priors = ConditionalPriorDict()

minimum_spacing = 0
t_min = 0
t_max = 2000
if n_models == 1:
    mean_priors[f'mean:amplitude_0'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e12, name='A')
    mean_priors[f'mean:sigma_0'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=10000, name='sigma')
    mean_priors[f'mean:t_0_0'] = bilby.core.prior.Uniform(minimum=t_min, maximum=t_max, name='t_0')
else:
    for ii in range(n_models):
        if ii == 0:
            mean_priors[f"mean:t_0_{ii}"] = Beta(minimum=t_min, maximum=t_max, alpha=1, beta=n_models, name=f"t_0_{ii}")
        else:
            mean_priors[f"mean:t_0_{ii}"] = QPOEstimation.prior.minimum.MinimumPrior(
                order=n_models - ii, minimum_spacing=minimum_spacing, minimum=t_min, maximum=t_max, name=f"t_0_{ii}")
        mean_priors[f'mean:amplitude_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e12, name=f'A_{ii}')
        mean_priors[f'mean:sigma_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=10000, name=f'sigma_{ii}')
        mean_priors[f"mean:t_0_{ii}"].__class__.__name__ = "QPOEstimation.prior.minimum.MinimumPrior"

priors.update(mean_priors)

# label = f'stabilised_gaussian_{n_models}'
# label = f'stabilised_gaussian_gp_{n_models}_{recovery_mode}'

# n_freds = 3
# label = f'fred_{n_freds}_{recovery_mode}'

# mean_model = get_n_component_fred_model(n_freds=n_freds)
# mean_model = get_n_component_stabilised_fred_model(n_freds=n_freds)
# mean_priors = get_fred_priors(n_freds=n_freds, t_min=0, t_max=2000, minimum_spacing=0)
# priors.update(mean_priors)
# priors['kernel:log_f'].minimum -= 0.2


# mean_model = np.mean(counts)
label = f'{n_models}_gaussian_{n_shos}_shos_only'
if likelihood_model == 'gaussian_process':
    likelihood = CeleriteLikelihood(kernel=kernel, mean_model=mean_model, fit_mean=True, t=times,
                                    y=counts, yerr=yerr)
else:
    likelihood = WindowedCeleriteLikelihood(kernel=kernel, mean_model=mean_model, fit_mean=True, t=times,
                                            y=counts, yerr=yerr)
    window_priors = get_window_priors(times=times)
    priors.update(window_priors)

# plt.plot(times, mean_model.get_value(times))
# plt.errorbar(times, counts, yerr=yerr, fmt=".k", capsize=0, label='data')
# plt.show()
# plt.clf()


# plt.plot(times, mean_model.get_value(times))
# plt.errorbar(times, counts, yerr=yerr, fmt=".k", capsize=0, label='data')
# plt.show()
# plt.clf()


# result = bilby.result.read_in_result(filename=f"{outdir}/results/{label}_result.json")
result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                           label=label, sampler='dynesty', nlive=nlive, sample='rslice',
                           resume=False, clean=True, use_ratio=use_ratio)
try:
    result.plot_corner(outdir=f"{outdir}/corner", truths=truths)
except Exception:
    result.plot_corner(outdir=f"{outdir}/corner")
# max_like_params = result.posterior.iloc[-1]
# plt.plot(times, counts)
# plt.plot(times, gauss(times, **max_like_params), color='red')
# for i in range(10):
#     params = result.posterior.iloc[np.random.randint(len(result.posterior))]
#     plt.plot(times, gauss(times, **params), alpha=0.2, color='red')

# plt.show()
# assert False
if plot:
    if recovery_mode in ["qpo", "pure_qpo", "general_qpo"]:
        try:
            try:
                frequency_samples = np.exp(np.array(result.posterior['kernel:log_f']))
            except Exception as e:
                frequency_samples = np.exp(np.array(result.posterior['kernel:terms[0]:log_f']))
            plt.hist(frequency_samples, bins="fd", density=True)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('normalised PDF')
            median = np.median(frequency_samples)
            percentiles = np.percentile(frequency_samples, [16, 84])
            plt.title(
                f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            plt.savefig(f"{outdir}/corner/{label}_frequency_posterior")
            plt.clf()
        except Exception as e:
            bilby.core.utils.logger.info(e)

    max_like_params = result.posterior.iloc[-1]
    for name, value in max_like_params.items():
        try:
            likelihood.gp.set_parameter(name=name, value=value)
        except ValueError:
            continue
        try:
            mean_model.set_parameter(name=name, value=value)
        except (ValueError, AttributeError):
            continue

    Path(f"{outdir}/fits/").mkdir(parents=True, exist_ok=True)
    taus = np.linspace(-400, 400, 1000)
    plt.plot(taus, likelihood.gp.kernel.get_value(taus))
    plt.xlabel('tau [s]')
    plt.ylabel('kernel')
    plt.savefig(f"{outdir}/fits/{label}_max_like_kernel")
    plt.clf()

    if likelihood_model == 'gaussian_process_windowed':
        plt.axvline(max_like_params['window_minimum'], color='cyan', label='start/end stochastic process')
        # plt.axvline(max_like_params['window_minimum'] + max_like_params['window_size'], color='cyan')
        plt.axvline(max_like_params['window_maximum'], color='cyan')
        # x = np.linspace(max_like_params['window_minimum'], max_like_params['window_minimum'] + max_like_params['window_size'], 5000)
        x = np.linspace(max_like_params['window_minimum'], max_like_params['window_maximum'], 5000)
        # windowed_indices = np.where(np.logical_and(max_like_params['window_minimum'] < t, t < max_like_params['window_minimum'] + max_like_params['window_size']))
        windowed_indices = np.where(
            np.logical_and(max_like_params['window_minimum'] < times, times < max_like_params['window_maximum']))
        likelihood.gp.compute(times[windowed_indices], yerr[windowed_indices])
        pred_mean, pred_var = likelihood.gp.predict(counts[windowed_indices], x, return_var=True)
        pred_std = np.sqrt(pred_var)
    else:
        x = np.linspace(times[0], times[-1], 5000)
        pred_mean, pred_var = likelihood.gp.predict(counts, x, return_var=True)
        pred_std = np.sqrt(pred_var)

    color = "#ff7f0e"
    plt.errorbar(times, counts, yerr=yerr, fmt=".k", capsize=0, label='data')
    plt.plot(x, pred_mean, color=color, label='Prediction')
    plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
                     edgecolor="none")
    if background_model != "mean":
        x = np.linspace(times[0], times[-1], 5000)
        for key, val in max_like_params.items():
            try:
                mean_model.set_parameter(key.replace('mean:', ''), val)
            except Exception:
                continue
        try:
            trend = mean_model.get_value(x)
        except AttributeError:
            trend = np.ones(len(x)) * mean_model
        plt.plot(x, trend, color='green', label='Mean')

    plt.xlabel("time [s]")
    plt.ylabel("data")
    plt.legend()
    plt.savefig(f"{outdir}/fits/{label}_max_like_fit")
    plt.show()
    plt.clf()

    psd_freqs = np.exp(np.linspace(np.log(band_minimum), np.log(band_maximum), 5000))
    psd = likelihood.gp.kernel.get_psd(psd_freqs * 2 * np.pi)

    plt.loglog(psd_freqs, psd, label='complete GP')
    for i, k in enumerate(likelihood.gp.kernel.terms):
        plt.loglog(psd_freqs, k.get_psd(psd_freqs * 2 * np.pi), "--", label=f'term {i}')

    plt.xlim(psd_freqs[0], psd_freqs[-1])
    plt.xlabel("f[Hz]")
    plt.ylabel("$S(f)$")
    plt.legend()
    plt.savefig(f"{outdir}/fits/{label}_psd")
    plt.clf()


# clean up
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png',  # '_corner.png',
                  '_dynesty.pickle', '_resume.pickle', '_result.json.old', '_samples.dat']:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
