from astropy.io import fits

import bilby
import matplotlib.pyplot as plt

from QPOEstimation.smoothing import two_sided_exponential_smoothing
from QPOEstimation.likelihood import PoissonLikelihoodWithBackground
from QPOEstimation.model.series import *
import sys

run_id = int(sys.argv[1])
period_number = int(sys.argv[2])

fits_data = fits.open('data/SGR_1806_20/event_4ms.lc.gz')
times = fits_data[1].data["TIME"]
counts = fits_data[1].data["COUNTS"]
times -= times[0]
tail = np.where(np.logical_and(times > 210, times < 650))
times = times[tail]
counts = counts[tail]
times -= times[0]


interpulse_periods = [(193.5, 196.5), (201.0, 204.0), (208.5, 211.5), (216.0, 219.0), (223.5, 226.5), (231.0, 234.0), (238.5, 242.0), (246.5, 249.5), (254.0, 257.0)]
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
priors['start_time'] = bilby.core.prior.Uniform(minimum=start, maximum=stop, name='start_time')
priors['amplitude'] = bilby.core.prior.LogUniform(minimum=0.01, maximum=10, name='amplitude')
priors['decay_time'] = bilby.core.prior.Uniform(minimum=-1, maximum=1, name='decay_time')
priors['frequency'] = bilby.core.prior.LogUniform(minimum=10, maximum=128, name='frequency')
priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='phase')


likelihood = PoissonLikelihoodWithBackground(x=t, y=c, func=zeroed_qpo_shot, background=truncated_background)
label = f'{run_id}'
result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir,
                           label=label, sampler='dynesty', nlive=300, resume=True)
result.plot_corner()

max_like_params = result.posterior.iloc[-1]
print(max_like_params)


plt.plot(t, c - truncated_background, label='background subtracted data')
plt.plot(t, zeroed_qpo_shot(t, background=truncated_background, **max_like_params), label='max_likelihood')
plt.legend()
plt.savefig(f"{outdir}/max_like_fit_{run_id}")
plt.clf()