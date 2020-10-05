import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path


from astropy.io import fits
import bilby
import celerite
from celerite import terms
from scipy.signal import periodogram
import stingray

import QPOEstimation
from QPOEstimation.prior.minimum import MinimumPrior
from QPOEstimation.stabilisation import bar_lev
from QPOEstimation.model.series import *
from QPOEstimation.likelihood import CeleriteLikelihood, QPOTerm, ZeroedQPOTerm, WhittleLikelihood, PoissonLikelihoodWithBackground, GrothLikelihood
import matplotlib
# matplotlib.use('Qt5Agg')

run_id = int(sys.argv[1])
period_number = int(sys.argv[2])
n_qpos = int(sys.argv[3])
model_id = int(sys.argv[4])

# run_id = 0
# period_number = 0
# n_qpos = 2
# model_id = 0

# candidate_id = int(sys.argv[1])
# seg_id = int(sys.argv[2])
# n_qpos = int(sys.argv[2])
# model_id = int(sys.argv[3])

# n_qpos = 1
# candidate_id = 3
# model_id = 0
# seg_id = 'test'


# injection_id = int(sys.argv[1])
# n_qpos = int(sys.argv[2])
# model_id = int(sys.argv[3])

# n_qpos = 0
# injection_id = 0
# model_id = 0

likelihood_models = ['gaussian_process', 'periodogram', 'poisson']
likelihood_model = likelihood_models[model_id]
candidates_run = False
injection_run = False
# band = 'test'
band = '5_16Hz'
# band = '64_128Hz'
# band = '16_32Hz'
# band = 'miller'
miller_band_bounds = [(16, 64), (60, 128), (60, 128), (16, 64), (60, 128), (60, 128), (16, 64), (16, 64), (60, 128),
                      (10, 32), (128, 256), (16, 64), (16, 64), (16, 64), (128, 256), (16, 64), (16, 64), (60, 128),
                      (60, 128), (60, 128), (60, 128), (16, 64), (32, 64)]

if band == 'miller':
    band_minimum = miller_band_bounds[candidate_id][0]
    band_maximum = miller_band_bounds[candidate_id][1]
else:
    band_minimum = 16
    band_maximum = 32
# band = f'64_128Hz'
band_minimum = 5
band_maximum = 16
# sampling_frequency = 4*band_maximum
sampling_frequency = 8 * band_maximum

if injection_run:
    data = np.loadtxt(f'injection_files/{str(injection_id).zfill(2)}_data.txt')
else:
    if likelihood_model in [likelihood_models[0], likelihood_models[2]]:
        # data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz.dat')
        data = np.loadtxt(f'data/detrend_counts_{sampling_frequency}Hz.dat')
        # data = np.loadtxt(f'data/sgr1806_256Hz.dat')
    else:
        data = np.loadtxt(f'data/sgr1806_1024Hz.dat')
    # times[0] = 2004 December 27 at 21:30:31.375 UTC
times = data[:, 0]
counts = data[:, 1]


if candidates_run:
    candidates = np.loadtxt(f'candidates_{band}.txt')
    start = candidates[candidate_id][0]
    stop = candidates[candidate_id][1]
    if band == 'miller':  # Miller et al. time segments are shifted by 16 s
        start += 20.0
        stop += 20.0
        # start += 20.0
        # stop += 20.0
    seglen = stop - start

    segment_length = stop - start
elif injection_run:
    start = -0.1
    stop = 1.1
else:
    pulse_period = 7.56  # see papers
    interpulse_periods = []
    for i in range(47):
        interpulse_periods.append((20.0 + i * pulse_period, 20.0 + (i + 1) * pulse_period))

    start = interpulse_periods[period_number][0]

    segment_length = 0.5
    # segment_step = 0.135  # Requires 56 steps
    segment_step = 0.27  # Requires 28 steps

    start = start + run_id * segment_step
    stop = start + segment_length

indices = np.where(np.logical_and(times > start, times < stop))
t = times[indices]
c = counts[indices]
# c = c.astype(int)

if candidates_run:
    if n_qpos == 0:
        outdir = f"sliding_window_{band}_candidates/no_qpo"
    elif n_qpos == 1:
        outdir = f"sliding_window_{band}_candidates/one_qpo"
    else:
        outdir = f"sliding_window_{band}_candidates/two_qpo"
elif injection_run:
    if n_qpos == 0:
        outdir = f"sliding_window_{band}_injections/no_qpo"
    elif n_qpos == 1:
        outdir = f"sliding_window_{band}_injections/one_qpo"
    else:
        outdir = f"sliding_window_{band}_injections/two_qpo"
else:
    if n_qpos == 0:
        outdir = f"sliding_window_{band}/period_{period_number}/no_qpo"
    elif n_qpos == 1:
        outdir = f"sliding_window_{band}/period_{period_number}/one_qpo"
    else:
        outdir = f"sliding_window_{band}/period_{period_number}/two_qpo"


priors = bilby.core.prior.PriorDict()
if likelihood_model == likelihood_models[0]:

    # stabilised_counts = bar_lev(c)
    stabilised_counts = c
    # print(np.std(stabilised_counts))
    stabilised_variance = np.ones(len(stabilised_counts))

    plt.plot(t, stabilised_counts)
    plt.show()

    # A non-periodic component
    Q = 1.0 / np.sqrt(2.0)
    w0 = 3.0
    S0 = np.var(stabilised_counts) / (w0 * Q)

    if n_qpos == 0:
        # kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))
        # kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))
        # kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))
        kernel = QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)
        # kernel = QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)
    elif n_qpos == 1:
        # kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))
        # kernel *= terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))
        kernel = QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)
        # kernel += QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)
        # kernel = ZeroedQPOTerm(log_a=0.1, log_c=-0.01, log_P=-3)
        # for i in range(1, n_qpos):
        #     kernel += QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)
    elif n_qpos == 2:
        kernel = QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3) \
                 + QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)

    params_dict = kernel.get_parameter_dict()
    print(params_dict)
    # mean_model = PolynomialMeanModel(a0=0, a1=0, a2=0, a3=0)#, a4=0, a5=0, a6=0, a7=0, a8=0, a9=0)
    gp = celerite.GP(kernel=kernel, mean=np.mean(stabilised_counts))#, fit_mean=True)  # , mean=np.mean(stabilised_counts))
    gp.compute(t, np.ones(len(t)))  # You always need to call compute once.


    print(gp.get_parameter_vector())

    if n_qpos == 0:
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='log_a')
        # priors['kernel:log_b'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='log_b')
        priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=10, name='log_b')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(band_minimum), name='log_c')
        priors['kernel:log_P'] = bilby.core.prior.DeltaFunction(peak=-2, name='log_P')
        # priors['kernel:log_S0'] = bilby.core.prior.Uniform(minimum=-10, maximum=15, name='log_S0')
        # priors['kernel:log_omega0'] = bilby.core.prior.Uniform(minimum=-10, maximum=np.log(band_maximum*np.pi*np.sqrt(2)), name='log_omega0')
        # priors['kernel:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1/np.sqrt(2)), name='log_Q')
    elif n_qpos == 1:
        # priors['mean:a0'] = bilby.core.prior.Uniform(minimum=0, maximum=50, name='mean:a0')
        # priors['mean:a1'] = bilby.core.prior.Uniform(minimum=-5, maximum=5, name='mean:a1')
        # priors['mean:a2'] = bilby.core.prior.Uniform(minimum=-2, maximum=2, name='mean:a2')
        # priors['mean:a3'] = bilby.core.prior.Uniform(minimum=-1, maximum=1, name='mean:a3')
        # priors['mean:a4'] = bilby.core.prior.Uniform(minimum=-50, maximum=50, name='mean:a4')
        # priors['mean:a5'] = bilby.core.prior.Uniform(minimum=-50, maximum=50, name='mean:a5')
        # priors['mean:a6'] = bilby.core.prior.Uniform(minimum=-50, maximum=50, name='mean:a6')
        # priors['mean:a7'] = bilby.core.prior.Uniform(minimum=-50, maximum=50, name='mean:a7')
        # priors['mean:a8'] = bilby.core.prior.Uniform(minimum=-50, maximum=50, name='mean:a8')
        # priors['mean:a9'] = bilby.core.prior.Uniform(minimum=-50, maximum=50, name='mean:a9')
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='log_a')
        # priors['kernel:log_b'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='log_b')
        priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='log_b')
        # priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(sampling_frequency), name='log_c')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(band_minimum), name='log_c')
        priors['kernel:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(band_maximum), maximum=-np.log(band_minimum), name='log_P')
        # priors['kernel:log_P'] = bilby.core.prior.Uniform(minimum=np.log(7.56/2), maximum=np.log(7.56*8), name='log_P')
        # priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='log_a')
        # priors['kernel:terms[1]:log_b'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='log_b')
        # priors['kernel:terms[1]:log_b'] = bilby.core.prior.DeltaFunction(peak=10, name='log_b')
        # priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(sampling_frequency), name='log_c')
        # priors['kernel:terms[1]:log_P'] = bilby.core.prior.DeltaFunction(peak=3, name='log_P')
        # priors['kernel:k1:log_S0'] = bilby.core.prior.Uniform(minimum=-10, maximum=15, name='k1:log_S0')
        # priors['kernel:k1:log_S0'] = bilby.core.prior.DeltaFunction(peak=0, name='k1:log_S0')
        # priors['kernel:k1:log_omega0'] = bilby.core.prior.Uniform(minimum=-10, maximum=np.log(band_maximum*np.pi*np.sqrt(2)), name='k1:log_omega0')
        # priors['kernel:k1:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1/np.sqrt(2)), name='k1:log_Q')
        # priors['kernel:log_S0'] = bilby.core.prior.Uniform(minimum=-10, maximum=15, name='k2:log_S0')
        # priors['kernel:log_omega0'] = bilby.core.prior.Uniform(minimum=np.log(2*np.pi*band_minimum), maximum=np.log(2*np.pi*band_maximum), name='k2:log_omega0')
        # priors['kernel:log_Q'] = bilby.core.prior.Uniform(minimum=np.log(1/np.sqrt(2)), maximum=10, name='k2:log_Q')
    elif n_qpos == 2:
        priors = bilby.core.prior.ConditionalPriorDict()
        priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[0]:log_a')
        priors['kernel:terms[0]:log_b'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='terms[0]:log_b')
        priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(sampling_frequency), name='terms[0]:log_c')
        priors['kernel:terms[0]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(16), maximum=-np.log(10), name='terms[0]:log_P')
        # priors['kernel:terms[0]:log_P'] = bilby.core.prior.Uniform(minimum=np.log(7.56*0.75), maximum=np.log(7.56*8), name='terms[0]:log_P')
        priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[1]:log_a')
        priors['kernel:terms[1]:log_b'] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name='terms[1]:log_b')
        priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[1]:log_c')
        priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(64), maximum=-np.log(16), name='terms[1]:log_P')
        # priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=np.log(7.56*0.25), maximum=np.log(7.56*0.75), name='terms[1]:log_P')
        # priors['kernel:terms[1]:log_P'] = MinimumPrior(minimum=-np.log(band_maximum), maximum=-np.log(band_minimum), name='terms[1]:log_P', reference_name='kernel:terms[0]:log_P', order=1)
        # priors['kernel:terms[1]:log_P']._required_variables = ['kernel:terms[0]:log_P']

    likelihood = CeleriteLikelihood(gp=gp, y=stabilised_counts)
    noise_term = celerite.terms.JitterTerm(log_sigma=0)
    noise_gp = celerite.GP(noise_term)
    noise_gp.compute(t=t, yerr=np.ones(len(t)))

elif likelihood_model == likelihood_models[1]:
    lc = stingray.Lightcurve(time=t, counts=c)
    ps = stingray.Powerspectrum(lc=lc, norm='leahy')
    frequencies = ps.freq
    powers = ps.power
    powers /= 2  # Groth norm
    noise_model = 'red_noise'
    # fs = 1/(t[1] - t[0])
    # frequencies, powers = periodogram(c, fs)
    frequency_mask = [True] * len(frequencies)
    # frequency_mask[0] = False
    plt.loglog(frequencies[frequency_mask], powers[frequency_mask])
    plt.show()
    plt.clf()
    priors = QPOEstimation.prior.psd.get_full_prior(noise_model, frequencies=frequencies)
    priors['alpha'] = bilby.core.prior.DeltaFunction(peak=2, name='alpha')
    priors['beta'] = bilby.core.prior.Uniform(minimum=1, maximum=10000, name='beta')
    # priors['beta'].maximum = 100000
    # priors['sigma'].maximum = 10
    priors['sigma'] = bilby.core.prior.DeltaFunction(peak=0)
    priors['width'].maximum = 10
    priors['width'].minimum = frequencies[1] - frequencies[0]
    priors['central_frequency'].maximum = band_maximum
    priors['central_frequency'].minimum = band_minimum
    priors['amplitude'] = bilby.core.prior.Uniform(minimum=1, maximum=10000)
    # priors['amplitude'].maximum = 10000
    if n_qpos == 0:
        priors['amplitude'] = bilby.core.prior.DeltaFunction(0.0, name='amplitude')
        priors['width'] = bilby.core.prior.DeltaFunction(1.0, name='width')
        priors['central_frequency'] = bilby.core.prior.DeltaFunction(1.0, name='central_frequency')
    likelihood = GrothLikelihood(frequencies=frequencies, periodogram=powers, noise_model=noise_model)
    # likelihood = WhittleLikelihood(frequencies=frequencies, periodogram=powers, noise_model=noise_model,
    #                                frequency_mask=[True]*len(frequencies))
elif likelihood_model == likelihood_models[2]:
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
        likelihood = PoissonLikelihoodWithBackground(x=t, y=c, func=sine_func, background=background_estimate)
        priors['f'] = bilby.core.prior.LogUniform(minimum=band_minimum, maximum=band_maximum, name='f')
        priors['amplitude'] = bilby.core.prior.LogUniform(minimum=0.01, maximum=100, name='amplitude')
        priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='phase')


if likelihood_model == likelihood_models[0]:
    if candidates_run:
        label = f"{candidate_id}"
    elif injection_run:
        label = f"{str(injection_id).zfill(2)}"
    else:
        label = f'{run_id}'
elif likelihood_model == likelihood_models[1]:
    if candidates_run:
        label = f"{candidate_id}_groth"
    elif injection_run:
        label = f"{str(injection_id).zfill(2)}_groth"
    else:
        label = f'{run_id}_groth'
elif likelihood_model == likelihood_models[2]:
    if candidates_run:
        label = f"{candidate_id}_poisson"
    elif injection_run:
        label = f"{str(injection_id).zfill(2)}_poisson"
    else:
        label = f'{run_id}_poisson'


# try:
# result = bilby.result.read_in_result(outdir=f"{outdir}/results", label=label)
# except Exception:
    # pass
result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                           label=label, sampler='dynesty', nlive=400, sample='rwalk',
                           resume=True, clean=True)

if candidates_run or injection_run:
# if True:
    result.plot_corner(outdir=f"{outdir}/corner")
    if likelihood_model == likelihood_models[0]:
        if n_qpos == 1:
            try:
                frequency_samples = []
                # for i, sample in enumerate(result.posterior.iloc):
                #     frequency_samples.append(np.exp(sample['kernel:log_omega0'])/2/np.pi)
                for i, sample in enumerate(result.posterior.iloc):
                    frequency_samples.append(1 / np.exp(sample[f'kernel:log_P']))

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
        elif n_qpos == 2:
            try:
                frequency_samples_0 = np.array(1 / np.exp(result.posterior.iloc[f'kernel:terms[0]:log_P']))
                frequency_samples_1 = np.array(1 / np.exp(result.posterior.iloc[f'kernel:terms[1]:log_P']))
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

        # idxs = np.where(np.logical_and(155 < t, t < 165))[0]
        # xs = t[idxs]
        x = t
        # x = np.linspace(155, 165, 25000)
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
        plt.savefig(f"{outdir}/fits/{label}_max_like_fit")
        # plt.show()
        plt.clf()

        psd_freqs = np.exp(np.linspace(np.log(1.0), np.log(128), 5000))
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

        # pred_mean, pred_var = gp.predict(stabilised_counts, t, return_var=True)
        # plt.scatter(t, stabilised_counts - pred_mean, label='residual')
        # plt.fill_between(t, 1, -1, color=color, alpha=0.3, edgecolor="none")
        # plt.xlabel("time [s]")
        # plt.ylabel("stabilised residuals")
        # plt.savefig(f"{outdir}/fits/{label}_max_like_fit_residuals")
        # plt.clf()
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
        plt.show()
    elif likelihood_model == likelihood_models[2]:
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
