import matplotlib.pyplot as plt
import os
import sys

from astropy.io import fits
import bilby
import celerite
from celerite import terms

from QPOEstimation.stabilisation import anscombe, bar_lev
from QPOEstimation.model.series import *
from QPOEstimation.likelihood import CeleriteLikelihood, QPOTerm

# run_id = int(sys.argv[1])
# period_number = int(sys.argv[2])
# n_qpos = int(sys.argv[3])

# run_id = 9
# period_number = 2
# n_qpos = 1

candidate_id = int(sys.argv[1])
n_qpos = int(sys.argv[2])

# n_qpos = 1
# candidate_id = 79


# data = np.loadtxt(f'data/sgr1806_256Hz.dat')
data = np.loadtxt(f'data/sgr1806_64Hz.dat')
times = data[:, 0]
counts = data[:, 1]

candidates = True

if candidates:
    candidates = np.loadtxt('candidates_below_16Hz.txt')
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

band = '16_32Hz'

if candidates:
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

priors = bilby.core.prior.PriorDict()

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
    priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(16), maximum=-np.log(5), name='terms[1]:log_P')
    # priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-4.15, maximum=-2.0, name='terms[1]:log_P')
elif n_qpos == 2:
    priors['kernel:terms[0]:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='terms[0]:log_S0')
    priors['kernel:terms[0]:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1 / np.sqrt(2)), name='terms[0]:log_Q')
    priors['kernel:terms[0]:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='terms[0]:log_omega0')
    priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[1]:log_a')
    priors['kernel:terms[1]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[1]:log_b')
    priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[1]:log_c')
    priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(16), maximum=-np.log(5),
                                                               name='terms[1]:log_P')
    priors['kernel:terms[2]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[2]:log_a')
    priors['kernel:terms[2]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[2]:log_b')
    priors['kernel:terms[2]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[2]:log_c')
    priors['kernel:terms[2]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(16), maximum=-np.log(5),
                                                               name='terms[2]:log_P')

likelihood = CeleriteLikelihood(gp=gp, y=stabilised_counts)
if candidates:
    label = f"{candidate_id}"
else:
    label = f'{run_id}'
# label = f'testing_{freq}'

result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir,
                           label=label, sampler='dynesty', nlive=1000, sample='rwalk',
                           resume=False, plot=False, clean=True)
# result = bilby.result.read_in_result(outdir=outdir, label=label)
# result.plot_corner()

# for term in [1, 2]:
#     try:
#         frequency_samples = []
#         for i, sample in enumerate(result.posterior.iloc):
#             frequency_samples.append(1 / np.exp(sample[f'kernel:terms[{term}]:log_P']))
#
#         plt.hist(frequency_samples, bins="fd", density=True)
#         plt.xlabel('frequency [Hz]')
#         plt.ylabel('normalised PDF')
#         median = np.median(frequency_samples)
#         percentiles = np.percentile(frequency_samples, [16, 84])
#         plt.title(
#             f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
#         plt.savefig(f"{outdir}/frequency_posterior_{label}_{term}")
#         plt.clf()
#     except Exception:
#         continue
#
# max_like_params = result.posterior.iloc[-1]
# for name, value in max_like_params.items():
#     try:
#         gp.set_parameter(name=name, value=value)
#     except ValueError:
#         continue
#
# x = np.linspace(t[0], t[-1], 5000)
# pred_mean, pred_var = gp.predict(stabilised_counts, x, return_var=True)
# pred_std = np.sqrt(pred_var)
# plt.legend()
#
# color = "#ff7f0e"
# plt.errorbar(t, stabilised_counts, yerr=stabilised_variance, fmt=".k", capsize=0, label='data')
# plt.plot(x, pred_mean, color=color, label='Prediction')
# plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
#                  edgecolor="none")
# plt.xlabel("time [s]")
# plt.ylabel("variance stabilised data")
# # plt.show()
# plt.savefig(f"{outdir}/max_like_fit_{label}")
# plt.clf()

# clean up
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png', '_corner.png',
                  '_dynesty.pickle', '_resume.pickle', '_samples.dat']:
    try:
        os.remove(f"{outdir}/{label}{extension}")
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
