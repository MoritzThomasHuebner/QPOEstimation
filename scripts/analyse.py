import argparse
import json
import os
import sys
from pathlib import Path

import bilby
import celerite
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import periodogram

import QPOEstimation
from QPOEstimation.likelihood import CeleriteLikelihood, QPOTerm, WhittleLikelihood, \
    GrothLikelihood, ExponentialTerm, ZeroedQPOTerm, TransientCeleriteLikelihood
from QPOEstimation.model.series import *

likelihood_models = ["gaussian_process", "gaussian_process_windowed", "periodogram", "poisson"]
modes = ["qpo", "white_noise", "red_noise", "zeroed_qpo", "mixed"]

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", default='sliding_window', choices=['sliding_window', 'multiple_windows', 'candidates', 'injection'])
    parser.add_argument("--sampling_frequency", default=None, type=int)
    parser.add_argument("--data_mode", choices=['normal', 'smoothed', 'smoothed_residual'], default='normal', type=str)
    parser.add_argument("--alpha", default=0.02, type=float)

    parser.add_argument("--period_number", default=0, type=int)
    parser.add_argument("--run_id", default=0, type=int)

    parser.add_argument("--candidate_id", default=0, type=int)
    parser.add_argument("--miller_candidates", default=False, type=bool)

    parser.add_argument("--injection_id", default=0, type=int)
    parser.add_argument("--injection_mode", default="qpo", choices=modes, type=str)

    parser.add_argument("--polynomial_max", default=1000, type=float)
    parser.add_argument("--min_log_a", default=-5, type=float)
    parser.add_argument("--max_log_a", default=15, type=float)
    parser.add_argument("--min_log_c", default=-6, type=float)

    parser.add_argument("--recovery_mode", default="qpo", choices=modes)
    parser.add_argument("--model", default="gaussian_process", choices=likelihood_models)
    parser.add_argument("--background_model", default="polynomial", choices=["polynomial", "exponential", "mean"])
    parser.add_argument("--periodogram_likelihood", default="whittle", choices=["whittle", "groth"])
    parser.add_argument("--periodogram_noise_model", default="red_noise", choices=["red_noise", "broken_power_law"])

    parser.add_argument("--band_minimum", default=10, type=int)
    parser.add_argument("--band_maximum", default=32, type=int)

    parser.add_argument("--segment_length", default=1.0, type=float)
    parser.add_argument("--segment_step", default=0.27, type=float)
    parser.add_argument("--nlive", default=150, type=int)
    parser.add_argument("--use_ratio", default=False, type=bool)

    parser.add_argument("--try_load", default=True, type=bool)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--plot", default=True, type=bool)
    args = parser.parse_args()

    run_mode = args.run_mode
    sampling_frequency = args.sampling_frequency
    data_mode = args.data_mode
    alpha = args.alpha

    period_number = args.period_number
    run_id = args.run_id

    candidate_id = args.candidate_id
    miller_candidates = args.miller_candidates

    polynomial_max = args.polynomial_max
    min_log_a = args.min_log_a
    max_log_a = args.max_log_a
    min_log_c = args.min_log_c

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
    use_ratio = args.use_ratio

    try_load = args.try_load
    resume = args.resume
    plot = args.plot
else:
    matplotlib.use('Qt5Agg')

    run_mode = 'sliding_window'
    # run_mode = 'multiple_windows'
    sampling_frequency = 256
    data_mode = 'smoothed_residual'
    # data_mode = 'normal'
    alpha = 0.02

    period_number = 13
    run_id = 26

    candidate_id = 3
    miller_candidates = False

    injection_id = 0
    injection_mode = "mixed"

    polynomial_max = 1000
    min_log_a = -5
    max_log_a = 5
    min_log_c = -5

    recovery_mode = "qpo"
    likelihood_model = "gaussian_process_windowed"
    background_model = "mean"
    periodogram_likelihood = "whittle"
    periodogram_noise_model = "red_noise"

    band_minimum = 5
    band_maximum = 64
    # segment_length = 7.56
    # segment_length = 2.268
    segment_length = 1.8
    # segment_length = 2.
    segment_step = 0.23625   # Requires 32 steps
    # segment_step = 0.54   # Requires 14 steps

    nlive = 150
    use_ratio = False

    try_load = True
    resume = False
    plot = True

    suffix = ""

pulse_period = 7.56  # see papers
n_pulse_periods = 47
time_offset = 20.0

if miller_candidates:
    miller_band_bounds = [(16, 64), (60, 128), (60, 128), (16, 64), (60, 128), (60, 128), (16, 64), (16, 64), (60, 128),
                          (10, 32), (128, 256), (16, 64), (16, 64), (16, 64), (128, 256), (16, 64), (16, 64), (60, 128),
                          (60, 128), (60, 128), (60, 128), (16, 64), (32, 64)]
    band_minimum = miller_band_bounds[candidate_id][0]
    band_maximum = miller_band_bounds[candidate_id][1]
    band = 'miller'
else:
    band = f'{band_minimum}_{band_maximum}Hz'

if sampling_frequency is None:
    if band_maximum <= 64:
        sampling_frequency = 256
    elif band_maximum <= 128:
        sampling_frequency = 512
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
        if miller_candidates:  # Miller et al. time segments are shifted by 20 s
            start += time_offset
            stop += time_offset
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
    indices = np.where(np.logical_and(times > start, times < stop))[0]
    t = times[indices]
    c = counts[indices]


priors = bilby.core.prior.PriorDict()
if likelihood_model in ["gaussian_process", "gaussian_process_windowed"]:
    if run_mode == 'injection' or data_mode in ['smoothed', 'smoothed_residual']:
        stabilised_counts = c
    else:
        stabilised_counts = bar_lev(c)

    stabilised_variance = np.ones(len(c))
    plt.errorbar(t, stabilised_counts, yerr=np.sqrt(stabilised_variance), fmt=".k", capsize=0, label='data')
    plt.show()
    plt.clf()

    if background_model == 'polynomial':
        if polynomial_max == 0:
            priors['mean:a0'] = 0
            priors['mean:a1'] = 0
            priors['mean:a2'] = 0
            priors['mean:a3'] = 0
            priors['mean:a4'] = 0
            fit_mean = False
        else:
            priors['mean:a0'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a0')
            priors['mean:a1'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a1')
            priors['mean:a2'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a2')
            priors['mean:a3'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a3')
            priors['mean:a4'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a4')
            fit_mean = True
        mean_model = PolynomialMeanModel(a0=0, a1=0, a2=0, a3=0, a4=0)
    else:
        mean_model = np.mean(stabilised_counts)
        fit_mean = False

    if recovery_mode == "white_noise":
        kernel = celerite.terms.JitterTerm(log_sigma=-20)
        priors['kernel:log_sigma'] = bilby.core.prior.DeltaFunction(peak=-20, name='log_sigma')
    elif recovery_mode == "qpo":
        kernel = QPOTerm(log_a=0.1, log_b=-10, log_c=-0.01, log_f=3)
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='log_a')
        priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='log_b')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(sampling_frequency*16), name='log_c')
        priors['kernel:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum), maximum=np.log(band_maximum), name='log_f')
    elif recovery_mode == "zeroed_qpo":
        kernel = ZeroedQPOTerm(log_a=0.1, log_c=-0.01, log_f=3)
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='log_a')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(sampling_frequency*16), name='log_c')
        priors['kernel:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum), maximum=np.log(band_maximum), name='log_f')
    elif recovery_mode == "red_noise":
        kernel = ExponentialTerm(log_a=0.1, log_c=-0.01)
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='log_a')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(sampling_frequency*16), name='log_c')
    elif recovery_mode == "mixed":
        kernel = QPOTerm(log_a=0.1, log_b=-10, log_c=-0.01, log_f=3) + ExponentialTerm(log_a=0.1, log_c=-0.01)
        # kernel = ZeroedQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + ExponentialTerm(log_a=0.1, log_c=-0.01)
        # kernel = ExponentialTerm(log_a=0.1, log_c=-0.01) + ExponentialTerm(log_a=0.1, log_c=-0.01)
        priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='terms[0]:log_a')
        priors['kernel:terms[0]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[0]:log_b')
        priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(sampling_frequency*16), name='terms[0]:log_c')
        # priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=3.2, name='terms[0]:log_c')
        priors['kernel:terms[0]:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum), maximum=np.log(band_maximum), name='terms[0]:log_f')
        priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='terms[1]:log_a')
        priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(sampling_frequency*16), name='terms[1]:log_c')
        # priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=3.2, maximum=np.log(sampling_frequency*16), name='terms[1]:log_c')
    else:
        raise ValueError('Recovery mode not defined')

    gp = celerite.GP(kernel=kernel, mean=mean_model, fit_mean=fit_mean)
    gp.compute(t, np.sqrt(stabilised_variance))
    if likelihood_model == "gaussian_process_windowed":
        priors['window_minimum'] = bilby.core.prior.Uniform(minimum=t[0], maximum=t[-1], name='window_minimum')
        priors['window_size'] = bilby.core.prior.Uniform(minimum=0, maximum=segment_length, name='window_size')
        likelihood = TransientCeleriteLikelihood(mean_model=mean_model, kernel=kernel, fit_mean=fit_mean, t=t, y=stabilised_counts)
    else:
        likelihood = CeleriteLikelihood(gp=gp, y=stabilised_counts)

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
                                       frequency_mask=[True]*len(frequencies))
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
    if run_mode == 'injection' and injection_mode == recovery_mode:
        try:
            result.plot_corner(outdir=f"{outdir}/corner", truths=truths)
        except Exception:
            result.plot_corner(outdir=f"{outdir}/corner")
    else:
        result.plot_corner(outdir=f"{outdir}/corner")

    if likelihood_model in ["gaussian_process", "gaussian_process_windowed"]:
        if recovery_mode in ["qpo", "zeroed_qpo", "mixed"]:
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
                plt.title(f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
                plt.savefig(f"{outdir}/corner/{label}_frequency_posterior")
                plt.clf()
            except Exception as e:
                bilby.core.utils.logger.info(e)

        max_like_params = result.posterior.iloc[-1]
        for name, value in max_like_params.items():
            try:
                gp.set_parameter(name=name, value=value)
            except ValueError:
                continue
            try:
                mean_model.set_parameter(name=name, value=value)
            except (ValueError, AttributeError):
                continue

        Path(f"{outdir}/fits/").mkdir(parents=True, exist_ok=True)
        taus = np.linspace(-0.5, 0.5, 1000)
        plt.plot(taus, gp.kernel.get_value(taus))
        plt.xlabel('tau [s]')
        plt.ylabel('kernel')
        plt.savefig(f"{outdir}/fits/{label}_max_like_kernel")
        plt.clf()

        x = np.linspace(t[0], t[-1], 5000)
        pred_mean, pred_var = gp.predict(stabilised_counts, x, return_var=True)
        pred_std = np.sqrt(pred_var)

        color = "#ff7f0e"
        plt.errorbar(t, stabilised_counts, yerr=np.sqrt(stabilised_variance), fmt=".k", capsize=0, label='data')
        plt.plot(x, pred_mean, color=color, label='Prediction')
        plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
                         edgecolor="none")
        if background_model != "mean":
            trend = mean_model.get_value(x)
            plt.plot(x, trend, color='green', label='Trend')

        plt.xlabel("time [s]")
        plt.ylabel("variance stabilised data")
        plt.legend()
        plt.savefig(f"{outdir}/fits/{label}_max_like_fit")
        plt.show()
        plt.clf()

        psd_freqs = np.exp(np.linspace(np.log(1.0), np.log(band_maximum), 5000))
        psd = gp.kernel.get_psd(psd_freqs*2*np.pi)

        plt.loglog(psd_freqs, psd, label='complete GP')
        for i, k in enumerate(gp.kernel.terms):
            plt.loglog(psd_freqs, k.get_psd(psd_freqs*2*np.pi), "--", label=f'term {i}')

        plt.xlim(psd_freqs[0], psd_freqs[-1])
        plt.xlabel("f[Hz]")
        plt.ylabel("$S(f)$")
        plt.legend()
        plt.savefig(f"{outdir}/fits/{label}_psd")
        plt.clf()

    elif likelihood_model == "periodogram":
        result.plot_corner(outdir=f"{outdir}/corner")
        if recovery_mode in ["qpo", "zeroed_qpo", "mixed"]:
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
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png', #'_corner.png',
                  '_dynesty.pickle', '_resume.pickle', '_result.json.old', '_samples.dat']:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
