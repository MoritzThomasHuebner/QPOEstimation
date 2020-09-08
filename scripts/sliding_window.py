import matplotlib.pyplot as plt
import os
import sys

from astropy.io import fits
import bilby
import celerite
from celerite import terms
from scipy.signal import periodogram

import QPOEstimation
from QPOEstimation.stabilisation import anscombe, bar_lev
from QPOEstimation.model.series import *
from QPOEstimation.likelihood import CeleriteLikelihood, QPOTerm, WhittleLikelihood, PoissonLikelihoodWithBackground

# run_id = int(sys.argv[1])
# period_number = int(sys.argv[2])
# n_qpos = int(sys.argv[3])

run_id = None
period_number = None
# n_qpos = 1

candidate_id = int(sys.argv[1])
n_qpos = int(sys.argv[2])
model_id = int(sys.argv[3])

# n_qpos = 1
# candidate_id = 0
# model_id = 2

likelihood_models = ['gaussian_process', 'periodogram', 'poisson']
likelihood_model = likelihood_models[model_id]
candidates_run = True
band = 'below_16Hz'

band_minimum = 5
band_maximum = 16

if likelihood_model in [likelihood_models[0], likelihood_models[2]]:
    data = np.loadtxt(f'data/sgr1806_64Hz.dat')
else:
    data = np.loadtxt(f'data/sgr1806_1024Hz.dat')
times = data[:, 0]
counts = data[:, 1]


if candidates_run:
    candidates = np.loadtxt(f'candidates_{band}.txt')
    start = candidates[candidate_id][0]
    stop = candidates[candidate_id][1]
    seglen = stop - start
    print(seglen)

    if seglen < 1:
        extend = 1 - seglen
        start -= extend/2
        stop += extend/2

    segment_length = stop - start
else:
    pulse_period = 7.56  # see papers
    interpulse_periods = []
    for i in range(47):
        interpulse_periods.append((10.0 + i * pulse_period, 10.0 + (i + 1) * pulse_period))

    start = interpulse_periods[period_number][0]

    segment_length = 1.0
    segment_step = 0.135  # Requires 56 steps

    start = start + run_id * segment_step
    stop = start + segment_length

indices = np.where(np.logical_and(times > start, times < stop))
t = times[indices]
c = counts[indices]
c = c.astype(int)

if candidates_run:
    if n_qpos == 0:
        outdir = f"sliding_window_{band}_candidates/no_qpo"
    elif n_qpos == 1:
        outdir = f"sliding_window_{band}_candidates/one_qpo"
    else:
        outdir = f"sliding_window_{band}_candidates/two_qpo"
else:
    if n_qpos == 0:
        outdir = f"sliding_window_{band}/period_{period_number}/no_qpo"
    elif n_qpos == 1:
        outdir = f"sliding_window_{band}/period_{period_number}/one_qpo"
    else:
        outdir = f"sliding_window_{band}/period_{period_number}/two_qpo"

stabilised_counts = bar_lev(c)
stabilised_variance = np.ones(len(stabilised_counts))

# A non-periodic component
Q = 1.0 / np.sqrt(2.0)
w0 = 3.0
S0 = np.var(stabilised_counts) / (w0 * Q)

kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))
for i in range(n_qpos):
    kernel += QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)

params_dict = kernel.get_parameter_dict()

gp = celerite.GP(kernel, mean=np.mean(stabilised_counts))
gp.compute(t, stabilised_variance)  # You always need to call compute once.


def sine_func(t, amplitude, f, phase, **kwargs):
    return amplitude * np.sin(2 * np.pi * f * t + phase)


priors = bilby.core.prior.PriorDict()
if likelihood_model == likelihood_models[0]:
    if n_qpos == 0:
        priors['kernel:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='log_S0')
        priors['kernel:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='log_omega0')
        priors['kernel:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1/np.sqrt(2)), name='log_Q')
    elif n_qpos == 1:
        priors['kernel:terms[0]:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='terms[0]:log_S0')
        priors['kernel:terms[0]:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='terms[0]:log_omega0')
        # priors['kernel:terms[0]:log_S0'] = bilby.core.prior.DeltaFunction(peak=-15, name='terms[0]:log_S0')
        # priors['kernel:terms[0]:log_omega0'] = bilby.core.prior.DeltaFunction(peak=-1, name='terms[0]:log_omega0')
        priors['kernel:terms[0]:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1 / np.sqrt(2)), name='terms[0]:log_Q')
        priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[1]:log_a')
        priors['kernel:terms[1]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[1]:log_b')
        priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[1]:log_c')
        priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(band_maximum), maximum=-np.log(band_minimum), name='terms[1]:log_P')
        # priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-4.15, maximum=-2.0, name='terms[1]:log_P')
    elif n_qpos == 2:
        priors['kernel:terms[0]:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='terms[0]:log_S0')
        priors['kernel:terms[0]:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1 / np.sqrt(2)), name='terms[0]:log_Q')
        priors['kernel:terms[0]:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='terms[0]:log_omega0')
        priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[1]:log_a')
        priors['kernel:terms[1]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[1]:log_b')
        priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[1]:log_c')
        priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(band_maximum), maximum=-np.log(band_minimum),
                                                                   name='terms[1]:log_P')
        priors['kernel:terms[2]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[2]:log_a')
        priors['kernel:terms[2]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[2]:log_b')
        priors['kernel:terms[2]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[2]:log_c')
        priors['kernel:terms[2]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(band_maximum), maximum=-np.log(band_minimum),
                                                                   name='terms[2]:log_P')

    likelihood = CeleriteLikelihood(gp=gp, y=stabilised_counts)
elif likelihood_model == likelihood_models[1]:
    noise_model = 'red_noise'

    fs = 1/(t[1] - t[0])
    frequencies, powers = periodogram(c, fs)
    frequency_mask = [True] * len(frequencies)
    frequency_mask[0] = False
    plt.loglog(frequencies[frequency_mask], powers[frequency_mask])
    plt.show()
    plt.clf()
    priors = QPOEstimation.prior.psd.get_full_prior(noise_model, frequencies=frequencies)
    priors['central_frequency'].maximum = band_maximum
    priors['central_frequency'].minimum = band_minimum
    if n_qpos == 0:
        priors['amplitude'] = bilby.core.prior.DeltaFunction(0.0, name='amplitude')
        priors['width'] = bilby.core.prior.DeltaFunction(1.0, name='width')
        priors['central_frequency'] = bilby.core.prior.DeltaFunction(1.0, name='central_frequency')
    likelihood = WhittleLikelihood(frequencies=frequencies, periodogram=powers,
                                   frequency_mask=frequency_mask, noise_model=noise_model)
elif likelihood_model == likelihood_models[2]:
    priors = bilby.core.prior.PriorDict()
    background_estimate = QPOEstimation.smoothing.two_sided_exponential_smoothing(counts, alpha=0.06)
    background_estimate = background_estimate[indices]
    likelihood = PoissonLikelihoodWithBackground(x=t, y=c, func=sine_func, background=background_estimate)
    priors['f'] = bilby.core.prior.LogUniform(minimum=band_minimum, maximum=band_maximum, name='f')
    priors['amplitude'] = bilby.core.prior.LogUniform(minimum=0.01, maximum=100, name='amplitude')
    priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='phase')

# label = f'testing_{freq}'

if likelihood_model == likelihood_models[0]:
    if candidates_run:
        label = f"{candidate_id}"
    else:
        label = f'{run_id}'
elif likelihood_model == likelihood_models[1]:
    if candidates_run:
        label = f"{candidate_id}_whittle"
    else:
        label = f'{run_id}_whittle'
elif likelihood_model == likelihood_models[2]:
    if candidates_run:
        label = f"{candidate_id}_poisson"
    else:
        label = f'{run_id}_poisson'


# try:
# result = bilby.result.read_in_result(outdir=outdir, label=label)
# except Exception:
result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                           label=label, sampler='dynesty', nlive=200, sample='rwalk',
                           resume=False, clean=True)
result.plot_corner(outdir=f"{outdir}/corner")
if likelihood_model == likelihood_models[0]:
    for term in [1, 2]:
        try:
            frequency_samples = []
            for i, sample in enumerate(result.posterior.iloc):
                frequency_samples.append(1 / np.exp(sample[f'kernel:terms[{term}]:log_P']))

            plt.hist(frequency_samples, bins="fd", density=True)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('normalised PDF')
            median = np.median(frequency_samples)
            percentiles = np.percentile(frequency_samples, [16, 84])
            plt.title(
                f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            plt.savefig(f"{outdir}/fits/frequency_posterior_{label}_{term}")
            plt.clf()
        except Exception:
            continue
    #
    max_like_params = result.posterior.iloc[-1]
    for name, value in max_like_params.items():
        try:
            gp.set_parameter(name=name, value=value)
        except ValueError:
            continue

    x = np.linspace(t[0], t[-1], 5000)
    pred_mean, pred_var = gp.predict(stabilised_counts, x, return_var=True)
    pred_std = np.sqrt(pred_var)
    plt.legend()

    color = "#ff7f0e"
    plt.errorbar(t, stabilised_counts, yerr=stabilised_variance, fmt=".k", capsize=0, label='data')
    plt.plot(x, pred_mean, color=color, label='Prediction')
    plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
                     edgecolor="none")
    plt.xlabel("time [s]")
    plt.ylabel("variance stabilised data")
    # plt.show()
    plt.savefig(f"{outdir}/fits/{label}_max_like_fit")
    plt.clf()

    pred_mean, pred_var = gp.predict(stabilised_counts, t, return_var=True)
    plt.plot(t - pred_mean, stabilised_counts, label='residual')
    plt.fill_between(t, 1, -1, color=color, alpha=0.3, edgecolor="none")
    plt.xlabel("time [s]")
    plt.ylabel("stabilised residuals")
    plt.savefig(f"{outdir}/fits/{label}_max_like_fit_residuals")
    plt.clf()
elif likelihood_model == likelihood_models[1]:
    if n_qpos > 0:
        frequency_samples = result.posterior['central_frequency']
        plt.hist(frequency_samples, bins="fd", density=True)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('normalised PDF')
        median = np.median(frequency_samples)
        percentiles = np.percentile(frequency_samples, [16, 84])
        plt.title(
            f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
        plt.savefig(f"{outdir}/corner/frequency_posterior_whittle_{label}")
        plt.clf()

    likelihood.parameters = result.posterior.iloc[-1]
    plt.loglog(frequencies[frequency_mask], powers[frequency_mask], label="Measured")
    plt.loglog(likelihood.frequencies, likelihood.model + likelihood.psd, color='r', label='max_likelihood')
    for i in range(10):
        likelihood.parameters = result.posterior.iloc[np.random.randint(len(result.posterior))]
        plt.loglog(likelihood.frequencies, likelihood.model + likelihood.psd, color='r', alpha=0.2)
    plt.legend()
    plt.savefig(f'{outdir}/fits/{label}_fitted_spectrum.png')
    plt.show()
elif likelihood_model == likelihood_models[2]:
    frequency_samples = result.posterior["f"]
    plt.hist(frequency_samples, bins="fd", density=True)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('normalised PDF')
    median = np.median(frequency_samples)
    percentiles = np.percentile(frequency_samples, [16, 84])
    plt.title(
        f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
    plt.savefig(f"{outdir}/corner/frequency_posterior_poisson_{label}")
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
# max_like_params = dict(amplitude=0, frequency=29, phase=0, mu=241.05, sigma=0.1, elevation=1, c_0=1, c_1=-0, c_2=1, c_3=0, c_4=-0.0)

# plt.plot(t, c, label='Data')
# plt.plot(t, sine_model(t, **max_like_params) + truncated_background, label='max_likelihood fit')
# plt.plot(t, truncated_background, label='background estimate')
# plt.xlabel("time [s]")
# plt.ylabel("counts")
# plt.legend()
# plt.savefig(f"{outdir}/max_like_fit_{label}")
# plt.clf()
