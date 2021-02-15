import argparse
import json
import os
import sys
from pathlib import Path

import bilby
import matplotlib
import matplotlib.pyplot as plt

import QPOEstimation
from QPOEstimation.likelihood import CeleriteLikelihood, WhittleLikelihood, \
    GrothLikelihood, WindowedCeleriteLikelihood, get_kernel
from QPOEstimation.model.celerite import PolynomialMeanModel
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


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", default='sliding_window', choices=run_modes)
    parser.add_argument("--sampling_frequency", default=None, type=int)
    parser.add_argument("--data_mode", choices=data_modes, default='normal', type=str)
    parser.add_argument("--alpha", default=0.02, type=float)

    parser.add_argument("--start_time", default=0., type=float)
    parser.add_argument("--end_time", default=1., type=float)

    parser.add_argument("--period_number", default=0, type=int)
    parser.add_argument("--run_id", default=0, type=int)

    parser.add_argument("--candidate_id", default=0, type=int)

    parser.add_argument("--injection_id", default=0, type=int)
    parser.add_argument("--injection_mode", default="qpo", choices=modes, type=str)

    parser.add_argument("--polynomial_max", default=1000, type=float)
    parser.add_argument("--min_log_a", default=-5, type=float)
    parser.add_argument("--max_log_a", default=15, type=float)
    parser.add_argument("--min_log_c", default=-6, type=float)
    parser.add_argument("--minimum_window_spacing", default=0, type=float)

    parser.add_argument("--recovery_mode", default="qpo", choices=modes)
    parser.add_argument("--model", default="gaussian_process", choices=likelihood_models)
    parser.add_argument("--background_model", default="polynomial", choices=background_models)
    parser.add_argument("--periodogram_likelihood", default="whittle", choices=["whittle", "groth"])
    parser.add_argument("--periodogram_noise_model", default="red_noise", choices=["red_noise", "broken_power_law"])

    parser.add_argument("--band_minimum", default=10, type=int)
    parser.add_argument("--band_maximum", default=32, type=int)

    parser.add_argument("--segment_length", default=1.0, type=float)
    parser.add_argument("--segment_step", default=0.27, type=float)
    parser.add_argument("--nlive", default=150, type=int)
    parser.add_argument("--use_ratio", default='False', type=str)

    parser.add_argument("--try_load", default='True', type=str)
    parser.add_argument("--resume", default='False', type=str)
    parser.add_argument("--plot", default='True', type=str)
    args = parser.parse_args()

    run_mode = args.run_mode
    sampling_frequency = args.sampling_frequency
    data_mode = args.data_mode
    alpha = args.alpha

    start_time = args.start_time
    end_time = args.end_time

    period_number = args.period_number
    run_id = args.run_id

    candidate_id = args.candidate_id

    polynomial_max = args.polynomial_max
    min_log_a = args.min_log_a
    max_log_a = args.max_log_a
    min_log_c = args.min_log_c
    minimum_window_spacing = args.minimum_window_spacing

    injection_id = args.injection_id
    injection_mode = args.injection_mode

    recovery_mode = args.recovery_mode
    likelihood_model = args.model
    background_model = args.background_model
    periodogram_likelihood = args.periodogram_likelihood
    periodogram_noise_model = args.periodogram_noise_model

    band_minimum = args.band_minimum
    band_maximum = args.band_maximum

    segment_length = args.segment_length
    segment_step = args.segment_step

    nlive = args.nlive
    use_ratio = boolean_string(args.use_ratio)

    try_load = boolean_string(args.try_load)
    resume = boolean_string(args.resume)
    plot = boolean_string(args.plot)
else:
    matplotlib.use('Qt5Agg')

    run_mode = 'sliding_window'
    # run_mode = 'select_time'
    sampling_frequency = 256
    data_mode = 'smoothed_residual'
    # data_mode = 'normal'
    alpha = 0.02

    start_time = 10
    end_time = 400

    period_number = 13
    run_id = 11

    candidate_id = 3

    injection_id = 2201
    injection_mode = "qpo"

    polynomial_max = 1000
    min_log_a = -5
    max_log_a = 5
    min_log_c = -5
    minimum_window_spacing = 0

    recovery_mode = "red_noise"
    likelihood_model = "gaussian_process_windowed"
    # background_model = "polynomial"
    background_model = "mean"
    periodogram_likelihood = "whittle"
    periodogram_noise_model = "red_noise"

    band_minimum = 5
    band_maximum = 64
    # segment_length = 7.56
    # segment_length = 2.268
    segment_length = 2.4
    # segment_length = 2.
    segment_step = 0.23625  # Requires 32 steps
    # segment_step = 0.54   # Requires 14 steps

    nlive = 100
    use_ratio = True

    try_load = True
    resume = False
    plot = True

    suffix = ""

pulse_period = 7.56  # see papers
n_pulse_periods = 47
time_offset = 20.0

band = f'{band_minimum}_{band_maximum}Hz'

if sampling_frequency is None:
    if band_maximum <= 64:
        sampling_frequency = 256
        alpha = 0.02
    elif band_maximum <= 128:
        sampling_frequency = 512
        alpha = 0.01
    else:
        sampling_frequency = 1024

if run_mode == 'injection':
    data = np.loadtxt(f'injection_files/{injection_mode}/{str(injection_id).zfill(2)}_data.txt')
    if injection_mode == recovery_mode:
        with open(f'injection_files/{injection_mode}/{str(injection_id).zfill(2)}_params.json', 'r') as f:
            truths = json.load(f)
    else:
        truths = {}
else:
    if data_mode == 'smoothed':
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_exp_smoothed_alpha_{alpha}.dat')
    elif data_mode == 'smoothed_residual':
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_exp_residual_alpha_{alpha}.dat')
    elif data_mode == 'blind_injection':
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_{data_mode}.dat')
    else:
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz.dat')

times = data[:, 0]
counts = data[:, 1]
outdir = 'outdir'
label = 'run'
start = times[0] - 0.1
stop = times[-1] + 0.1

starts = []
stops = []

if run_mode == 'multiple_windows':
    interpulse_periods = []
    for i in range(n_pulse_periods):
        interpulse_periods.append((time_offset + i * pulse_period, time_offset + (i + 1) * pulse_period))

    for i in range(20):
        starts.append(interpulse_periods[period_number + i][0] + run_id * segment_step)
        stops.append(starts[-1] + segment_length)
    outdir = f"{run_mode}_{band}_{data_mode}/period_{period_number}/{recovery_mode}"
    label = f'{run_id}_{likelihood_model}'

    indices = np.array([], dtype=int)
    for start, stop in zip(starts, stops):
        indices = np.append(indices, np.where(np.logical_and(times > start, times < stop))[0])

    t = times[indices]
    c = counts[indices]
else:
    if run_mode == 'candidates':
        candidates = np.loadtxt(f'candidates/candidates_{band}_{data_mode}.txt')
        start = candidates[candidate_id][0]
        stop = start + segment_length
        segment_length = stop - start

        outdir = f"{run_mode}_{band}_{data_mode}/{recovery_mode}"
        label = f"{candidate_id}_{likelihood_model}"
    elif run_mode == 'injection':
        outdir = f"{run_mode}_{band}_{data_mode}_{injection_mode}/{recovery_mode}"
        label = f"{str(injection_id).zfill(2)}_{likelihood_model}"
    elif run_mode == 'sliding_window':
        interpulse_periods = []
        for i in range(n_pulse_periods):
            interpulse_periods.append((time_offset + i * pulse_period, time_offset + (i + 1) * pulse_period))
        start = interpulse_periods[period_number][0] + run_id * segment_step
        stop = start + segment_length

        outdir = f"{run_mode}_{band}_{data_mode}/period_{period_number}/{recovery_mode}"
        label = f'{run_id}_{likelihood_model}'
    elif run_mode == 'select_time':
        start = start_time
        stop = end_time
        outdir = f"{run_mode}_{band}_{data_mode}/{start_time}_{end_time}/{recovery_mode}"
        label = f'{likelihood_model}'

    indices = np.where(np.logical_and(times > start, times < stop))[0]
    t = times[indices]
    c = counts[indices]

# Move center of light curve to 0
t -= t[0]
t -= t[-1]/2

priors = bilby.core.prior.PriorDict()
if likelihood_model in ["gaussian_process", "gaussian_process_windowed"]:
    if run_mode == 'injection' or data_mode in ['smoothed', 'smoothed_residual', 'blind_injection']:
        stabilised_counts = c
    else:
        stabilised_counts = bar_lev(c)

    stabilised_variance = np.ones(len(c))
    plt.errorbar(t, stabilised_counts, yerr=np.sqrt(stabilised_variance), fmt=".k", capsize=0, label='data')
    plt.show()
    plt.clf()

    if background_model == 'polynomial':
        fit_mean = (polynomial_max != 0)
        mean_priors = get_polynomial_prior(polynomial_max=polynomial_max)
        priors.update(mean_priors)
        mean_model = PolynomialMeanModel(a0=0, a1=0, a2=0, a3=0, a4=0)
    else:
        fit_mean = False
        mean_model = np.mean(stabilised_counts)

    kernel_priors = get_kernel_prior(kernel_type=recovery_mode, min_log_a=min_log_a, max_log_a=max_log_a,
                                     min_log_c=min_log_c, band_minimum=band_minimum, band_maximum=band_maximum)
    priors.update(kernel_priors)
    kernel = get_kernel(kernel_type=recovery_mode)

    if likelihood_model == "gaussian_process_windowed":
        window_priors = get_window_priors(times=t)
        priors.update(window_priors)

        def window_conversion_func(sample):
            sample['window_maximum'] = sample['window_minimum'] + sample['window_size']
            if injection_mode in ['qpo', 'pure_qpo', 'mixed', 'general_qpo']:
                sample = decay_constrain_conversion_function(sample=sample)
            return sample

        if recovery_mode in ['qpo', 'pure_qpo', 'mixed', 'general_qpo']:
            priors.conversion_function = decay_constrain_conversion_function

        likelihood = WindowedCeleriteLikelihood(mean_model=mean_model, kernel=kernel, fit_mean=fit_mean, t=t,
                                                y=stabilised_counts, yerr=np.sqrt(stabilised_variance))
    else:
        if recovery_mode in ['qpo', 'pure_qpo', 'mixed', 'general_qpo']:
            priors.conversion_function = decay_constrain_conversion_function
        likelihood = CeleriteLikelihood(kernel=kernel, mean_model=mean_model, fit_mean=fit_mean, t=t,
                                        y=stabilised_counts, yerr=np.sqrt(stabilised_variance))

elif likelihood_model == "periodogram":
    lc = stingray.Lightcurve(time=t, counts=c)
    ps = stingray.Powerspectrum(lc=lc, norm="leahy")
    frequencies = ps.freq
    powers = ps.power
    if periodogram_likelihood == "groth":
        powers /= 2

    frequency_mask = [True] * len(frequencies)
    plt.loglog(frequencies[frequency_mask], powers[frequency_mask])
    plt.show()
    plt.clf()
    priors = QPOEstimation.prior.psd.get_full_prior(periodogram_noise_model, frequencies=frequencies)
    priors['beta'] = bilby.core.prior.Uniform(minimum=1, maximum=100000, name='beta')
    priors['sigma'].maximum = 10
    priors['width'].maximum = 10
    priors['width'].minimum = frequencies[1] - frequencies[0]
    priors['central_frequency'].maximum = band_maximum
    priors['central_frequency'].minimum = band_minimum
    priors['amplitude'] = bilby.core.prior.LogUniform(minimum=1, maximum=10000)
    if recovery_mode == "white_noise":
        priors['amplitude'] = bilby.core.prior.DeltaFunction(0.0, name='amplitude')
        priors['width'] = bilby.core.prior.DeltaFunction(1.0, name='width')
        priors['central_frequency'] = bilby.core.prior.DeltaFunction(1.0, name='central_frequency')
    if periodogram_likelihood == "whittle":
        likelihood = WhittleLikelihood(frequencies=frequencies, periodogram=powers, noise_model=periodogram_noise_model,
                                       frequency_mask=[True] * len(frequencies))
    else:
        priors['sigma'] = bilby.core.prior.DeltaFunction(peak=0)
        likelihood = GrothLikelihood(frequencies=frequencies, periodogram=powers, noise_model=periodogram_noise_model)
else:
    raise ValueError("Likelihood model not defined")

result = None
if try_load:
    try:
        result = bilby.result.read_in_result(outdir=f"{outdir}/results", label=label)
    except Exception:
        pass

if result is None:
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                               label=label, sampler='dynesty', nlive=nlive, sample='rwalk',
                               resume=resume, use_ratio=use_ratio)

if plot:
    try:
        result.plot_corner(outdir=f"{outdir}/corner", truths=truths)
    except Exception:
        result.plot_corner(outdir=f"{outdir}/corner")
    else:
        result.plot_corner(outdir=f"{outdir}/corner")

    if likelihood_model in ["gaussian_process", "gaussian_process_windowed"]:
        if recovery_mode in ["qpo", "pure_qpo", "mixed", "general_qpo"]:
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
        taus = np.linspace(-0.5, 0.5, 1000)
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
                np.logical_and(max_like_params['window_minimum'] < t, t < max_like_params['window_maximum']))
            likelihood.gp.compute(t[windowed_indices], np.sqrt(stabilised_variance[windowed_indices]))
            pred_mean, pred_var = likelihood.gp.predict(stabilised_counts[windowed_indices], x, return_var=True)
            pred_std = np.sqrt(pred_var)
        else:
            x = np.linspace(t[0], t[-1], 5000)
            pred_mean, pred_var = likelihood.gp.predict(stabilised_counts, x, return_var=True)
            pred_std = np.sqrt(pred_var)

        color = "#ff7f0e"
        plt.errorbar(t, stabilised_counts, yerr=np.sqrt(stabilised_variance), fmt=".k", capsize=0, label='data')
        plt.plot(x, pred_mean, color=color, label='Prediction')
        plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
                         edgecolor="none")
        if background_model != "mean":
            x = np.linspace(t[0], t[-1], 5000)
            trend = mean_model.get_value(x)
            plt.plot(x, trend, color='green', label='Trend')

        plt.xlabel("time [s]")
        plt.ylabel("variance stabilised data")
        plt.legend()
        plt.savefig(f"{outdir}/fits/{label}_max_like_fit")
        plt.show()
        plt.clf()

        psd_freqs = np.exp(np.linspace(np.log(1.0), np.log(band_maximum), 5000))
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

    elif likelihood_model == "periodogram":
        result.plot_corner(outdir=f"{outdir}/corner")
        if recovery_mode in ["qpo", "mixed"]:
            frequency_samples = result.posterior['central_frequency']
            plt.hist(frequency_samples, bins="fd", density=True)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('normalised PDF')
            median = np.median(frequency_samples)
            percentiles = np.percentile(frequency_samples, [16, 84])
            plt.title(
                f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            plt.savefig(f"{outdir}/corner/{label}_frequency_posterior")
            plt.clf()

        likelihood.parameters = result.posterior.iloc[-1]
        plt.loglog(frequencies[frequency_mask], powers[frequency_mask], label="Measured")
        plt.loglog(likelihood.frequencies, likelihood.model + likelihood.psd, color='r', label='max_likelihood')
        for i in range(10):
            likelihood.parameters = result.posterior.iloc[np.random.randint(len(result.posterior))]
            plt.loglog(likelihood.frequencies, likelihood.model + likelihood.psd, color='r', alpha=0.2)
        plt.legend()
        Path(f"{outdir}/fits/").mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{outdir}/fits/{label}_fitted_spectrum.png')
        # plt.show()

# clean up
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png',  # '_corner.png',
                  '_dynesty.pickle', '_resume.pickle', '_result.json.old', '_samples.dat']:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
