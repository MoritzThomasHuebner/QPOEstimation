import matplotlib.pyplot as plt
import sys

from astropy.io import fits
import bilby
import celerite
from celerite import terms

from QPOEstimation.stabilisation import anscombe, bar_lev
from QPOEstimation.model.series import *
from QPOEstimation.prior.slabspike import SlabSpikePrior
from QPOEstimation.likelihood import CeleriteLikelihood, QPOTerm

run_id = int(sys.argv[1])
period_number = int(sys.argv[2])

# run_id = 0
# period_number = 0

fits_data = fits.open('data/SGR_1806_20/event_4ms.lc.gz')
times = fits_data[1].data["TIME"]
counts = fits_data[1].data["COUNTS"]
times -= times[0]
tail = np.where(np.logical_and(times > 210, times < 650))
times = times[tail]
counts = counts[tail]
times -= times[0]


pulse_period = 7.56  # see papers
interpulse_periods = []
for i in range(44):
    # interpulse_periods.append((193.3 + i*pulse_period, 196.3 + i*pulse_period))
    interpulse_periods.append((16.0 + i*pulse_period, 16.0 + (i + 1)*pulse_period))

# concat_times = np.array([])
# concat_counts = np.array([])
# period_freqs = np.array([])
# period_powers = np.array([])
# period_times = np.array([])
# period_counts = np.array([])

# for period in interpulse_periods:
#     start = period[0]
#     stop = period[1]
#     t = times[np.where(np.logical_and(times > start, times < stop))]
#     c = counts[np.where(np.logical_and(times > start, times < stop))]
#     concat_times = np.append(concat_times, t)
#     concat_counts = np.append(concat_counts, c)
#     lc = stingray.Lightcurve(t - t[0], c)
#     ps = stingray.Powerspectrum(lc)
#     period_freqs = np.append(period_freqs, ps.freq)
#     period_powers = np.append(period_powers, ps.power)


start = interpulse_periods[period_number][0]

segment_length = 1.0
segment_step = pulse_period / 38.0

start = start + run_id*segment_step
stop = start + segment_length


# background_start = start - 0.4
# background_stop = stop + 0.4
indices = np.where(np.logical_and(times > start, times < stop))
t = times[indices]
c = counts[indices]
c = c.astype(int)


outdir = f"sliding_window/period_{period_number}/two_qpo"

stabilised_counts = bar_lev(c)
stabilised_variance = np.ones(len(stabilised_counts))

# A non-periodic component
Q = 1.0 / np.sqrt(2.0)
w0 = 3.0
S0 = np.var(stabilised_counts) / (w0 * Q)

kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0)) \
         + QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)\
         + QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)

params_dict = kernel.get_parameter_dict()

gp = celerite.GP(kernel, mean=np.mean(counts))
gp.compute(t, stabilised_variance)  # You always need to call compute once.

# print("Initial log likelihood: {0}".format(gp.log_likelihood(stabilised_counts)))
# print("parameter_dict:\n{0}\n".format(gp.get_parameter_dict()))
# print("parameter_names:\n{0}\n".format(gp.get_parameter_names()))
# print("parameter_vector:\n{0}\n".format(gp.get_parameter_vector()))
# print("parameter_bounds:\n{0}\n".format(gp.get_parameter_bounds()))
# print(gp.get_parameter_dict())

# def log_P_fraction(sample):
#     res = deepcopy(sample)
#     res['log_P_fraction'] = sample['kernel:terms[1]:log_P']/sample['kernel:terms[2]:log_P']
#     return res


priors = bilby.core.prior.PriorDict()
# priors = bilby.core.prior.PriorDict(conversion_function=log_P_fraction)
# priors['kernel:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=40, name='log_S0')
# priors['kernel:log_omega0'] = bilby.core.prior.Uniform(minimum=-40, maximum=15, name='log_omega0')
# priors['kernel:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1/np.sqrt(2)), name='log_Q')

priors['kernel:terms[0]:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=40, name='terms[0]:log_S0')
priors['kernel:terms[0]:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1/np.sqrt(2)), name='terms[0]:log_Q')
priors['kernel:terms[0]:log_omega0'] = bilby.core.prior.Uniform(minimum=-40, maximum=15, name='terms[0]:log_omega0')
priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[1]:log_a')
priors['kernel:terms[1]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[1]:log_b')
priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[1]:log_c')
priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-4.16, maximum=-2.0, name='terms[1]:log_P')

priors['kernel:terms[2]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[2]:log_a')
priors['kernel:terms[2]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[2]:log_b')
priors['kernel:terms[2]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[2]:log_c')
priors['kernel:terms[2]:log_P'] = bilby.core.prior.Uniform(minimum=-4.85, maximum=-4.16, name='terms[2]:log_P')
# priors['log_P_fraction'] = bilby.core.prior.Constraint(minimum=0, maximum=1)

likelihood = CeleriteLikelihood(gp=gp, y=stabilised_counts)
label = f'{run_id}'

result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir,
                           label=label, sampler='dynesty', nlive=300, sample='rwalk', resume=True)
# result.plot_corner()
# result = bilby.result.read_in_result(outdir=outdir, label=label)

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
        plt.title(f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
        plt.savefig(f"{outdir}/frequency_posterior_{label}_{term}")
        plt.clf()
    except Exception:
        continue


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

# import matplotlib
# matplotlib.use('Qt5Agg')

color = "#ff7f0e"
plt.errorbar(t, stabilised_counts, yerr=stabilised_variance, fmt=".k", capsize=0, label='data')
plt.plot(x, pred_mean, color=color, label='Prediction')
plt.fill_between(x, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3,
                 edgecolor="none")
plt.xlabel("time [s]")
plt.ylabel("variance stabilised data")
# plt.show()
plt.savefig(f"{outdir}/max_like_fit_{label}")
plt.clf()

# max_like_params = dict(amplitude=0, frequency=29, phase=0, mu=241.05, sigma=0.1, elevation=1, c_0=1, c_1=-0, c_2=1, c_3=0, c_4=-0.0)

# plt.plot(t, c, label='Data')
# plt.plot(t, sine_model(t, **max_like_params) + truncated_background, label='max_likelihood fit')
# plt.plot(t, truncated_background, label='background estimate')
# plt.xlabel("time [s]")
# plt.ylabel("counts")
# plt.legend()
# plt.savefig(f"{outdir}/max_like_fit_{label}")
# plt.clf()