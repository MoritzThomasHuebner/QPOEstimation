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

recovery_mode = "pure_qpo"
likelihood_model = "gaussian_process_windowed"
background_model = "gaussian"


nlive = 300
use_ratio = False

plot = True

suffix = ""


data = np.loadtxt(f'data/SolarFlare/120704187_ctime_lc.txt')
# outdir = f"SolarFlare/120704187/"
outdir = f"SolarFlare/120704187/"

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

# times = times[60:250]
# counts = counts[60:250]
# yerr = yerr[60:250]
# times = times[155:345]
# counts = counts[155:345]
# yerr = yerr[155:345]
# times = times[250:440]
# counts = counts[250:440]
# yerr = yerr[250:440]
times = times[60:]
counts = counts[60:]
yerr = yerr[60:]
# times = times[60:440]
# counts = counts[60:440]
# yerr = yerr[60:440]

plt.plot(times, counts)
plt.show()
# yerr /= max_rate




start = times[0]
stop = times[-1]

dt = times[1] - times[0]

sampling_frequency = 1/dt

band_minimum = 1/(stop - start)
band_maximum = sampling_frequency/2
band = f'{band_minimum}_{band_maximum}Hz'

priors = bilby.core.prior.ConditionalPriorDict()



# kernel = get_kernel(kernel_type=recovery_mode) + get_kernel(recovery_mode) + get_kernel(recovery_mode)
# priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name='kernel:terms[0]:log_a')
# priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=-15, maximum=np.log(band_maximum), name='kernel:terms[0]:log_c')
# priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name='kernel:terms[1]:log_a')
# priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-15, maximum=np.log(band_maximum), name='kernel:terms[1]:log_c')
# priors['kernel:terms[2]:log_a'] = bilby.core.prior.Uniform(minimum=-15, maximum=30, name='kernel:terms[2]:log_a')
# priors['kernel:terms[2]:log_c'] = bilby.core.prior.Uniform(minimum=-15, maximum=np.log(band_maximum), name='kernel:terms[2]:log_c')
#
# priors['kernel:terms[0]:log_f'] = Beta(minimum=np.log(band_minimum), maximum=np.log(band_maximum), alpha=1, beta=3, name='kernel:terms[0]:log_f')
# priors['kernel:terms[1]:log_f'] = MinimumPrior(minimum=np.log(band_minimum), maximum=np.log(band_maximum), order=2, minimum_spacing=0, reference_name='kernel:terms[0]:log_f', name='kernel:terms[1]:log_f')
# priors['kernel:terms[2]:log_f'] = MinimumPrior(minimum=np.log(band_minimum), maximum=np.log(band_maximum), order=1, minimum_spacing=0, reference_name='kernel:terms[1]:log_f', name='kernel:terms[2]:log_f')
#
# priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name='decay_constraint')
# priors._resolve_conditions()



min_log_c = np.log(1/(times[-1] - times[0]))
kernel = get_kernel(kernel_type=recovery_mode)
kernel_priors = get_kernel_prior(kernel_type=recovery_mode, min_log_a=-15, max_log_a=30.,
                                 min_log_c=min_log_c, band_minimum=band_minimum, band_maximum=band_maximum)
priors.update(kernel_priors)

# priors.conversion_function = decay_constrain_conversion_function

# mean_model = np.mean(counts)
# kernel = celerite.terms.JitterTerm(log_sigma=-20)
# priors["kernel:log_sigma"] = bilby.core.prior.DeltaFunction(peak=-20, name='log_sigma')


n_models = 5
label = f'gaussian_{n_models}'


# mean_model = 0
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
            mean_priors[f"mean:t_0_{ii}"] = Beta(minimum=t_min, maximum=t_max, alpha=1, beta=n_models, name=f"mean:t_0_{ii}")
        else:
            mean_priors[f"mean:t_0_{ii}"] = QPOEstimation.prior.minimum.MinimumPrior(
                order=n_models - ii, minimum_spacing=minimum_spacing, minimum=t_min, maximum=t_max, name=f"mean:t_0_{ii}")
        mean_priors[f'mean:amplitude_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e12, name=f'A_{ii}')
        mean_priors[f'mean:sigma_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=10000, name=f'sigma_{ii}')
        mean_priors[f"mean:t_0_{ii}"].__class__.__name__ = "QPOEstimation.prior.minimum.MinimumPrior"

priors.update(mean_priors)


# n_freds = 1
# label = f"fred_{n_freds}_{recovery_mode}"
# label = f"fred_{n_freds}_{recovery_mode}"

# mean_model = get_n_component_fred_model(n_freds=n_freds)

# minimum_spacing = 0
# t_min = 0
# t_max = 2000
#
# mean_priors = get_fred_priors(n_freds=n_freds, t_min=t_min, t_max=t_max, minimum_spacing=minimum_spacing)
# priors.update(mean_priors)

priors._resolve_conditions()


# label = f'stabilised_gaussian_{n_models}'
# label = f'stabilised_gaussian_gp_{n_models}_{recovery_mode}'

# n_freds = 3
# label = f'fred_{n_freds}_{recovery_mode}'

# mean_model = get_n_component_fred_model(n_freds=n_freds)
# mean_model = get_n_component_stabilised_fred_model(n_freds=n_freds)
# priors.update(mean_priors)
# priors['kernel:log_f'].minimum -= 0.2


# mean_model = np.mean(counts)
# label = f'{n_models}_gaussian_{n_shos}_shos_only'
if likelihood_model == 'gaussian_process':
    likelihood = CeleriteLikelihood(kernel=kernel, mean_model=mean_model, fit_mean=True, t=times,
                                    y=counts, yerr=yerr)
else:
    likelihood = WindowedCeleriteLikelihood(kernel=kernel, mean_model=mean_model, fit_mean=True, t=times,
                                            y=counts, yerr=yerr)
    window_priors = get_window_priors(times=times)
    priors.update(window_priors)


priors._resolve_conditions()
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
max_like_params = result.posterior.iloc[-1]


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
                np.savetxt(f"{outdir}/corner/freq_samples_0.txt", frequency_samples)

            plt.hist(frequency_samples, bins="fd", density=True)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('normalised PDF')
            median = np.median(frequency_samples)
            percentiles = np.percentile(frequency_samples, [16, 84])
            plt.title(
                f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            plt.savefig(f"{outdir}/corner/{label}_frequency_posterior")
            plt.clf()

            try:
                frequency_samples = np.exp(np.array(result.posterior['kernel:terms[1]:log_f']))
                np.savetxt(f"{outdir}/corner/freq_samples_1.txt", frequency_samples)
                plt.hist(frequency_samples, bins="fd", density=True)
                plt.xlabel('frequency [Hz]')
                plt.ylabel('normalised PDF')
                median = np.median(frequency_samples)
                percentiles = np.percentile(frequency_samples, [16, 84])
                plt.title(
                    f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
                plt.savefig(f"{outdir}/corner/{label}_frequency_posterior_1")
                plt.clf()
            except Exception as e:
                pass

            try:
                frequency_samples = np.exp(np.array(result.posterior['kernel:terms[2]:log_f']))
                np.savetxt(f"{outdir}/corner/freq_samples_2.txt", frequency_samples)
                plt.hist(frequency_samples, bins="fd", density=True)
                plt.xlabel('frequency [Hz]')
                plt.ylabel('normalised PDF')
                median = np.median(frequency_samples)
                percentiles = np.percentile(frequency_samples, [16, 84])
                plt.title(
                    f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
                plt.savefig(f"{outdir}/corner/{label}_frequency_posterior_2")
                plt.clf()
            except Exception as e:
                pass


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
        plt.axvline(max_like_params['window_maximum'], color='cyan')
        x = np.linspace(max_like_params['window_minimum'], max_like_params['window_maximum'], 5000)
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
    # plt.errorbar(times, counts, yerr=yerr, fmt=".k", capsize=0, label='data')
    plt.plot(times, counts, label='data')
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
    # plt.ylabel("data")
    plt.ylabel("photon counts")
    plt.legend()
    plt.tight_layout()
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
