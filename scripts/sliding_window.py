from astropy.io import fits

import bilby
import matplotlib.pyplot as plt

from QPOEstimation.smoothing import two_sided_exponential_smoothing
from QPOEstimation.likelihood import PoissonLikelihoodWithBackground
from QPOEstimation.model.series import *
from QPOEstimation.prior.slabspike import SlabSpikePrior
import sys

# run_id = int(sys.argv[1])
# period_number = int(sys.argv[2])
run_id = 23
period_number = 6

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
for i in range(9):
    interpulse_periods.append((193.3 + i*pulse_period, 196.3 + i*pulse_period))

concat_times = np.array([])
concat_counts = np.array([])
period_freqs = np.array([])
period_powers = np.array([])
period_times = np.array([])
period_counts = np.array([])

for period in interpulse_periods:
    start = period[0]
    stop = period[1]
    t = times[np.where(np.logical_and(times > start, times < stop))]
    c = counts[np.where(np.logical_and(times > start, times < stop))]
    concat_times = np.append(concat_times, t)
    concat_counts = np.append(concat_counts, c)
    lc = stingray.Lightcurve(t - t[0], c)
    ps = stingray.Powerspectrum(lc)
    period_freqs = np.append(period_freqs, ps.freq)
    period_powers = np.append(period_powers, ps.power)


start = interpulse_periods[period_number][0]

segment_length = 0.3
segment_step = 0.1

start = start + run_id*segment_step
stop = start + segment_length


background_start = start - 0.4
background_stop = stop + 0.4
indices = np.where(np.logical_and(times > start, times < stop))
t = times[indices]
c = counts[indices]
c = c.astype(int)

indices = np.where(np.logical_and(times > background_start, times < background_stop))
background_t = times[indices]
background_c = counts[indices]
background_c = background_c.astype(int)
alpha = 0.02
background = two_sided_exponential_smoothing(background_c, alpha)

outdir = f"sliding_window/period_{period_number}"

# truncate background
truncated_background_start = np.where(background_t == t[0])[0][0]
truncated_background_stop = np.where(background_t == t[-1])[0][0]
truncated_background = background[truncated_background_start:truncated_background_stop + 1]
plt.plot(t, c)
plt.plot(t, truncated_background, label='background')
plt.savefig(f"{outdir}/background_{run_id}")
plt.clf()


priors = bilby.core.prior.PriorDict()
# priors['start_time'] = bilby.core.prior.Uniform(minimum=start, maximum=stop, name='start_time')
# priors['peak_time'] = bilby.core.prior.Uniform(minimum=start, maximum=stop, name='start_time')
# priors['decay_time'] = bilby.core.prior.LogUniform(minimum=1/256, maximum=2, name='decay_time')
# priors['mu'] = bilby.core.prior.Uniform(minimum=start-0.2, maximum=stop+0.2, name='mu')
# priors['sigma'] = bilby.core.prior.Uniform(minimum=1/128, maximum=0.6, name='sigma')
# priors['mu_1'] = bilby.core.prior.Uniform(minimum=start-0.2, maximum=start+0.3, name='mu_2')
# priors['sigma_1'] = bilby.core.prior.Uniform(minimum=1/128, maximum=0.6, name='sigma_1')
# priors['mu_2'] = bilby.core.prior.Uniform(minimum=start+0.3, maximum=stop+0.2, name='mu_1')
# priors['sigma_2'] = bilby.core.prior.Uniform(minimum=1/128, maximum=0.6, name='sigma_2')
priors['amplitude'] = bilby.core.prior.Uniform(minimum=0.10, maximum=100, name='amplitude')
priors['frequency'] = bilby.core.prior.LogUniform(minimum=10, maximum=128, name='frequency')
priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='phase')
priors['elevation'] = bilby.core.prior.LogUniform(minimum=0.01, maximum=100, name='elevation')
# priors['balance'] = bilby.core.prior.LogUniform(minimum=0.01, maximum=100, name='balance')
# priors['c_0'] = SlabSpikePrior(minimum=-10, maximum=10, spike_loc=0, spike_height=0.0, name='c_0')
# priors['c_1'] = SlabSpikePrior(minimum=-10, maximum=10, spike_loc=0, spike_height=0.9999, name='c_1')
# priors['c_2'] = SlabSpikePrior(minimum=-10, maximum=10, spike_loc=0, spike_height=0.9999, name='c_2')
# priors['c_3'] = SlabSpikePrior(minimum=-10, maximum=10, spike_loc=0, spike_height=0.9999, name='c_3')
# priors['c_4'] = SlabSpikePrior(minimum=-10, maximum=10, spike_loc=0, spike_height=0.9999, name='c_4')
# priors['c_5'] = SlabSpikePrior(minimum=-10, maximum=10, spike_loc=0, spike_height=0.9999, name='c_5')
# priors['c_6'] = SlabSpikePrior(minimum=-10, maximum=10, spike_loc=0, spike_height=0.9999, name='c_6')
# priors['c_7'] = SlabSpikePrior(minimum=-10, maximum=10, spike_loc=0, spike_height=0.9999, name='c_7')
# priors['c_8'] = SlabSpikePrior(minimum=-10, maximum=10, spike_loc=0, spike_height=0.9999, name='c_8')
# priors['c_0'] = bilby.core.prior.DeltaFunction(peak=1, name='c_0')
# priors['c_1'] = bilby.core.prior.DeltaFunction(peak=0, name='c_1')
# priors['c_2'] = bilby.core.prior.DeltaFunction(peak=0, name='c_2')
# priors['c_3'] = bilby.core.prior.DeltaFunction(peak=0, name='c_3')
# priors['c_4'] = bilby.core.prior.DeltaFunction(peak=0, name='c_4')
# priors['c_5'] = bilby.core.prior.DeltaFunction(peak=0, name='c_5')
# priors['c_6'] = bilby.core.prior.DeltaFunction(peak=0, name='c_6')
# priors['c_7'] = bilby.core.prior.DeltaFunction(peak=0, name='c_7')
# priors['c_8'] = bilby.core.prior.DeltaFunction(peak=0, name='c_8')
# priors['c_1'] = bilby.core.prior.DeltaFunction(peak=0, name='c_1')
# priors['c_2'] = bilby.core.prior.DeltaFunction(peak=0, name='c_2')
# priors['c_3'] = bilby.core.prior.DeltaFunction(peak=0, name='c_3')
# priors['c_4'] = bilby.core.prior.DeltaFunction(peak=0, name='c_4')
# priors['t_start'] = bilby.core.prior.Uniform(minimum=start-0.2, maximum=stop+0.2, name='t_start')
# priors['t_stop'] = bilby.core.prior.Uniform(minimum=start-0.2, maximum=stop+0.2, name='t_stop')
# priors['alpha'] = bilby.core.prior.Uniform(minimum=0, maximum=1, name='alpha')
# amplitude, frequency, phase, t_start, t_stop, elevation, alpha
# mu, sigma, amplitude, frequency, phase, elevation
# mu, sigma, amplitude, frequency, phase, elevation, c_0, c_1, c_2

likelihood = PoissonLikelihoodWithBackground(x=t, y=c, func=elevated_sine, background=truncated_background)
label = f'{run_id}'
result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir,
                           label=label, sampler='dynesty', nlive=300, sample='rwalk', resume=False)
result.plot_corner()

max_like_params = result.posterior.iloc[-1]
# max_like_params = dict(amplitude=0, frequency=29, phase=0, mu=241.05, sigma=0.1, elevation=1, c_0=1, c_1=-0, c_2=1, c_3=0, c_4=-0.0)

plt.plot(t, c - truncated_background, label='background subtracted data')
plt.plot(t, elevated_sine(t, **max_like_params), label='max_likelihood')
plt.legend()
plt.savefig(f"{outdir}/max_like_fit_{label}")
plt.clf()