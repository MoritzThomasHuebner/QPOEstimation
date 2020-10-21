import os
import sys
from pathlib import Path
import argparse

import bilby
import celerite
import matplotlib
import matplotlib.pyplot as plt

import QPOEstimation
from QPOEstimation.likelihood import CeleriteLikelihood, QPOTerm, WhittleLikelihood, \
    PoissonLikelihoodWithBackground, GrothLikelihood
from QPOEstimation.model.series import *

likelihood_models = ["gaussian_process", "periodogram", "poisson"]

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_run", default=False, type=bool)
    parser.add_argument("--miller_candidates", default=False, type=bool)
    parser.add_argument("--injection_run", default=False, type=bool)
    parser.add_argument("--run_id", default=0, type=int)
    parser.add_argument("--period_number", default=0, type=int)
    parser.add_argument("--n_qpos", default=0, type=int)
    parser.add_argument("--candidate_id", default=0, type=int)
    parser.add_argument("--injection_id", default=0, type=int)
    parser.add_argument("--model", default="gaussian_process", choices=likelihood_models)
    parser.add_argument("--band_minimum", default=10, type=int)
    parser.add_argument("--band_maximum", default=32, type=int)
    parser.add_argument("--segment_length", default=1.0, type=float)
    parser.add_argument("--segment_step", default=0.27, type=float)
    parser.add_argument("--background_model", default="polynomial", choices=["polynomial", "exponential", None])
    parser.add_argument("--periodogram_likelihood", default="whittle", choices=["whittle", "groth"])
    parser.add_argument("--periodogram_noise_model", default="red_noise", choices=["red_noise", "broken_power_law"])
    parser.add_argument("--nlive", default=150, type=int)
    parser.add_argument("--try_load", default=False, type=bool)
    parser.add_argument("--plot", default=False, type=bool)
    parser.add_argument("--suffix", default="", type=str)
    args = parser.parse_args()
    candidates_run = args.candidates_run
    miller_candidates = args.miller_candidates
    injection_run = args.injection_run
    run_id = args.run_id
    period_number = args.period_number
    n_qpos = args.n_qpos
    candidate_id = args.candidate_id
    injection_id = args.injection_id
    likelihood_model = args.model
    band_minimum = args.band_minimum
    band_maximum = args.band_maximum
    segment_length = args.segment_length
    segment_step = args.segment_step
    background_model = args.background_model
    periodogram_likelihood = args.periodogram_likelihood
    periodogram_noise_model = args.periodogram_noise_model
    nlive = args.nlive
    try_load = args.try_load
    plot = args.plot
    suffix = args.suffix
else:
    matplotlib.use('Qt5Agg')
    candidates_run = True
    miller_candidates = False
    injection_run = False
    period_number = 0
    run_id = 5
    n_qpos = 1
    candidate_id = 1
    injection_id = None
    band_minimum = 10
    band_maximum = 40
    likelihood_model = 'gaussian_process'
    segment_length = 1.0
    segment_step = 0.27   # Requires 28 steps
    background_model = 'polynomial'
    periodogram_likelihood = "whittle"
    periodogram_noise_model = "red_noise"
    nlive = 150
    try_load = False
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

# if band_maximum <= 16:
#     sampling_frequency = 128
# el
if band_maximum <= 64:
    sampling_frequency = 256
elif band_maximum <= 128:
    sampling_frequency = 512
else:
    sampling_frequency = 1024

if injection_run:
    data = np.loadtxt(f'injection_files/{str(injection_id).zfill(2)}_data.txt')
else:
    if likelihood_model in ['gaussian_process', 'poisson']:
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz.dat')
    else:
        data = np.loadtxt(f'data/sgr1806_1024Hz.dat')
        # times[0] = 2004 December 27 at 21:30:31.375 UTC

times = data[:, 0]
counts = data[:, 1]

if candidates_run:
    candidates = np.loadtxt(f'candidates/candidates_{band}{suffix}.txt')
    start = candidates[candidate_id][0]
    stop = start + segment_length
    if band == 'miller':  # Miller et al. time segments are shifted by 20 s
        start += time_offset
        stop += time_offset
    segment_length = stop - start
elif injection_run:
    start = -0.1
    stop = 1.1
else:
    interpulse_periods = []
    for i in range(n_pulse_periods):
        interpulse_periods.append((time_offset + i * pulse_period, time_offset + (i + 1) * pulse_period))
    start = interpulse_periods[period_number][0] + run_id * segment_step
    stop = start + segment_length

indices = np.where(np.logical_and(times > start, times < stop))
t = times[indices]
c = counts[indices]

if candidates_run:
    if n_qpos == 0:
        outdir = f"sliding_window_{band}{suffix}_candidates/no_qpo"
    elif n_qpos == 1:
        outdir = f"sliding_window_{band}{suffix}_candidates/one_qpo"
    else:
        outdir = f"sliding_window_{band}{suffix}_candidates/two_qpo"

    if likelihood_model == "gaussian_process":
        label = f"{candidate_id}"
    elif likelihood_model == "periodogram":
        label = f"{candidate_id}_{periodogram_likelihood}"
    else:
        label = f"{candidate_id}_poisson"
elif injection_run:
    if n_qpos == 0:
        outdir = f"sliding_window_{band}{suffix}_injections/no_qpo"
    elif n_qpos == 1:
        outdir = f"sliding_window_{band}{suffix}_injections/one_qpo"
    else:
        outdir = f"sliding_window_{band}{suffix}_injections/two_qpo"

    if likelihood_model == "gaussian_process":
        label = f"{str(injection_id).zfill(2)}"
    elif likelihood_model == "periodogram":
        label = f"{str(injection_id).zfill(2)}_{periodogram_likelihood}"
    else:
        label = f"{str(injection_id).zfill(2)}_poisson"
else:
    if n_qpos == 0:
        outdir = f"sliding_window_{band}{suffix}/period_{period_number}/no_qpo"
    elif n_qpos == 1:
        outdir = f"sliding_window_{band}{suffix}/period_{period_number}/one_qpo"
    else:
        outdir = f"sliding_window_{band}{suffix}/period_{period_number}/two_qpo"

    if likelihood_model == "gaussian_process":
        label = f'{run_id}'
    elif likelihood_model == "periodogram":
        label = f'{run_id}_{periodogram_likelihood}'
    else:
        label = f'{run_id}_poisson'


def conversion_function(sample):
    out_sample = deepcopy(sample)
    out_sample['decay_constraint'] = out_sample['kernel:log_c'] - out_sample['kernel:log_f']
    return out_sample


priors = bilby.core.prior.PriorDict()
if likelihood_model == "gaussian_process":
    stabilised_counts = bar_lev(c)
    stabilised_variance = np.ones(len(c))
    # from copy import deepcopy
    # stabilised_counts = deepcopy(c)
    # stabilised_variance = deepcopy(c)
    # stabilised_variance[np.where(stabilised_variance == 0)] = 1
    plt.errorbar(t, stabilised_counts, yerr=np.sqrt(stabilised_variance), fmt=".k", capsize=0, label='data')
    plt.show()
    plt.clf()

    if background_model == 'polynomial':
        priors['mean:a0'] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name='mean:a0')
        priors['mean:a1'] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name='mean:a1')
        priors['mean:a2'] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name='mean:a2')
        priors['mean:a3'] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name='mean:a3')
        priors['mean:a4'] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name='mean:a4')
        mean_model = PolynomialMeanModel(a0=0, a1=0, a2=0, a3=0, a4=0)
        fit_mean = True
    elif background_model == 'exponential':
        priors['mean:tau'] = bilby.core.prior.LogUniform(minimum=0.3, maximum=1.0, name='tau')
        priors['mean:offset'] = bilby.core.prior.LogUniform(minimum=1, maximum=50, name='offset')
        mean_model = ExponentialStabilisedMeanModel(tau=0, offset=0)
        fit_mean = True
    else:
        mean_model = np.mean(stabilised_counts)
        fit_mean = False

    if n_qpos == 0:
        kernel = celerite.terms.JitterTerm(log_sigma=-20)
        # priors['kernel:log_sigma'] = bilby.core.prior.DeltaFunction(peak=-20, name='log_sigma')
        priors['kernel:log_sigma'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='log_sigma')
    elif n_qpos == 1:
        kernel = QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_f=3)
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='log_a')
        priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='log_b')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(sampling_frequency), name='log_c')
        priors['kernel:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum), maximum=np.log(band_maximum), name='log_f')
        priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=-0.5, name='decay_constraint')
        priors.conversion_function = conversion_function
    elif n_qpos == 2:
        kernel = QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_f=3) \
                 + QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_f=3)
        priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[0]:log_a')
        priors['kernel:terms[0]:log_b'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='terms[0]:log_b')
        priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(sampling_frequency), name='terms[0]:log_c')
        priors['kernel:terms[0]:log_f'] = bilby.core.prior.Uniform(minimum=np.log(20), maximum=np.log(40), name='terms[0]:log_f')
        priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[1]:log_a')
        priors['kernel:terms[1]:log_b'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='terms[1]:log_b')
        priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[1]:log_c')
        priors['kernel:terms[1]:log_f'] = bilby.core.prior.Uniform(minimum=np.log(40), maximum=np.log(80), name='terms[1]:log_f')
    else:
        raise ValueError

    gp = celerite.GP(kernel=kernel, mean=mean_model, fit_mean=fit_mean)
    gp.compute(t, np.sqrt(stabilised_variance))  # You always need to call compute once.
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
    # plt.show()
    plt.clf()
    priors = QPOEstimation.prior.psd.get_full_prior(periodogram_noise_model, frequencies=frequencies)
    priors['beta'] = bilby.core.prior.Uniform(minimum=1, maximum=100000, name='beta')
    priors['sigma'].maximum = 10
    priors['width'].maximum = 10
    priors['width'].minimum = frequencies[1] - frequencies[0]
    priors['central_frequency'].maximum = band_maximum
    priors['central_frequency'].minimum = band_minimum
    priors['amplitude'] = bilby.core.prior.LogUniform(minimum=1, maximum=10000)
    if n_qpos == 0:
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
    if injection_run:
        if n_qpos == 0:
            priors['tau'] = bilby.core.prior.LogUniform(minimum=0.3, maximum=1.0, name='tau')
            priors['offset'] = bilby.core.prior.LogUniform(minimum=1, maximum=50, name='offset')
            priors['amplitude'] = bilby.core.prior.DeltaFunction(peak=0, name='amplitude')
            priors['mu'] = bilby.core.prior.DeltaFunction(peak=0.5, name='mu')
            priors['sigma'] = bilby.core.prior.DeltaFunction(peak=1, name='sigma')
            priors['frequency'] = bilby.core.prior.DeltaFunction(peak=1, name='frequency')
            priors['phase'] = bilby.core.prior.DeltaFunction(peak=0, name='phase')
        elif n_qpos == 1:
            priors['tau'] = bilby.core.prior.LogUniform(minimum=0.3, maximum=1.0, name='tau')
            priors['offset'] = bilby.core.prior.LogUniform(minimum=1, maximum=50, name='offset')
            priors['amplitude'] = bilby.core.prior.LogUniform(minimum=2, maximum=20, name='amplitude')
            priors['mu'] = bilby.core.prior.Uniform(minimum=0.3, maximum=0.7, name='mu')
            priors['sigma'] = bilby.core.prior.LogUniform(minimum=0.05, maximum=0.15, name='sigma')
            priors['frequency'] = bilby.core.prior.LogUniform(minimum=10, maximum=64, name='frequency')
            priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=2 * np.pi, name='phase')
        likelihood = bilby.core.likelihood.PoissonLikelihood(
            x=t, y=c, func=QPOEstimation.model.series.sine_gaussian_with_background)
    else:
        def sine_func(t, amplitude, f, phase, **kwargs):
            return amplitude * np.sin(2 * np.pi * f * t + phase)
        background_estimate = QPOEstimation.smoothing.two_sided_exponential_smoothing(counts, alpha=0.06)
        background_estimate = background_estimate[indices]
        priors['f'] = bilby.core.prior.LogUniform(minimum=band_minimum, maximum=band_maximum, name='f')
        priors['amplitude'] = bilby.core.prior.LogUniform(minimum=0.01, maximum=100, name='amplitude')
        priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='phase')
        likelihood = PoissonLikelihoodWithBackground(x=t, y=c, func=sine_func, background=background_estimate)

result = None
if try_load:
    try:
        result = bilby.result.read_in_result(outdir=f"{outdir}/results", label=label)
    except Exception:
        pass

if result is None:
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                               label=label, sampler='dynesty', nlive=nlive, sample='rwalk',
                               resume=True, clean=True)


if plot:
    result.plot_corner(outdir=f"{outdir}/corner")
    if likelihood_model == "gaussian_process":
        if n_qpos == 1:
            try:
                frequency_samples = np.exp(np.array(result.posterior['kernel:log_f']))
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
        elif n_qpos == 2:
            try:
                frequency_samples_0 = np.array(np.exp(result.posterior.iloc[f'kernel:terms[0]:log_f']))
                frequency_samples_1 = np.array(np.exp(result.posterior.iloc[f'kernel:terms[1]:log_f']))
                for samples, mode in zip([frequency_samples_0, frequency_samples_1], [0, 1]):
                    plt.hist(samples, bins="fd", density=True)
                    plt.xlabel('frequency [Hz]')
                    plt.ylabel('normalised PDF')
                    median = np.median(samples)
                    percentiles = np.percentile(samples, [16, 84])
                    plt.title(
                        f"{np.mean(samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
                    plt.savefig(f"{outdir}/corner/{label}_frequency_posterior_{mode}")
                    plt.clf()
            except Exception as e:
                bilby.core.utils.logger.info(e)

        max_like_params = result.posterior.iloc[-1]
        for name, value in max_like_params.items():
            try:
                gp.set_parameter(name=name, value=value)
            except ValueError:
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
        plt.legend()

        color = "#ff7f0e"
        plt.errorbar(t, stabilised_counts, yerr=np.sqrt(stabilised_variance), fmt=".k", capsize=0, label='data')
        plt.plot(x, pred_mean, color=color, label='Prediction')
        plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
                         edgecolor="none")
        plt.xlabel("time [s]")
        plt.ylabel("variance stabilised data")
        plt.savefig(f"{outdir}/fits/{label}_max_like_fit")
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

        pred_mean, pred_var = gp.predict(stabilised_counts, t, return_var=True)
        plt.scatter(t, stabilised_counts - pred_mean, label='residual')
        plt.fill_between(t, 1, -1, color=color, alpha=0.3, edgecolor="none")
        plt.xlabel("time [s]")
        plt.ylabel("stabilised residuals")
        plt.savefig(f"{outdir}/fits/{label}_max_like_fit_residuals")
        plt.clf()
    elif likelihood_model == "periodogram":
        if n_qpos > 0:
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
    elif likelihood_model == "poisson":
        if injection_run:
            frequency_samples = result.posterior["frequency"]
            plt.hist(frequency_samples, bins="fd", density=True)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('normalised PDF')
            median = np.median(frequency_samples)
            percentiles = np.percentile(frequency_samples, [16, 84])
            plt.title(
                f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            plt.savefig(f"{outdir}/corner/{label}_frequency_posterior")
            plt.clf()

            plt.plot(t, c, label='measured')
            max_like_params = result.posterior.iloc[-1]
            plt.plot(t, QPOEstimation.model.series.sine_gaussian_with_background(t, **max_like_params),
                     color='r', label='max_likelihood')
            for i in range(10):
                parameters = result.posterior.iloc[np.random.randint(len(result.posterior))]
                plt.plot(t, QPOEstimation.model.series.sine_gaussian_with_background(t, **parameters),
                         color='r', alpha=0.2)
            plt.xlabel("time [s]")
            plt.ylabel("counts")
            plt.legend()
            Path(f"{outdir}/fits/").mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{outdir}/fits/{label}_max_like_fit.png')
            plt.clf()

            plt.plot(t, c - QPOEstimation.model.series.sine_gaussian_with_background(t, **max_like_params), label='residual')
            plt.fill_between(t, np.sqrt(c), -np.sqrt(c), color='orange', alpha=0.3,
                             edgecolor="none", label='1 sigma uncertainty')
            plt.xlabel("time [s]")
            plt.ylabel("residuals")
            plt.legend()
            plt.savefig(f'{outdir}/fits/{label}_max_like_fit_residuals.png')
            plt.clf()
        else:
            frequency_samples = result.posterior["f"]
            plt.hist(frequency_samples, bins="fd", density=True)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('normalised PDF')
            median = np.median(frequency_samples)
            percentiles = np.percentile(frequency_samples, [16, 84])
            plt.title(
                f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            plt.savefig(f"{outdir}/corner/{label}_frequency_posterior")
            plt.clf()

            plt.plot(t, c, label='measured')
            max_like_params = result.posterior.iloc[-1]
            plt.plot(t, sine_func(t, **max_like_params) + background_estimate, color='r', label='max_likelihood')
            for i in range(10):
                parameters = result.posterior.iloc[np.random.randint(len(result.posterior))]
                plt.plot(t, sine_func(t, **parameters) + background_estimate, color='r', alpha=0.2)
            plt.xlabel("time [s]")
            plt.ylabel("counts")
            plt.legend()
            Path(f"{outdir}/fits/").mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{outdir}/fits/{label}_max_like_fit.png')
            plt.clf()

            plt.plot(t, c - sine_func(t, **max_like_params) - background_estimate, label='residual')
            plt.fill_between(t, np.sqrt(c), -np.sqrt(c), color='orange', alpha=0.3,
                             edgecolor="none", label='1 sigma uncertainty')
            plt.xlabel("time [s]")
            plt.ylabel("residuals")
            plt.legend()
            plt.savefig(f'{outdir}/fits/{label}_max_like_fit_residuals.png')
            plt.clf()


# clean up
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png', #'_corner.png',
                  '_dynesty.pickle', '_resume.pickle', '_result.json.old', '_samples.dat']:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
