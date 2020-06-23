import matplotlib

from QPOEstimation.functions import qpo_shot, burst_envelope

matplotlib.use("Qt5Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
from QPOEstimation.prior import *

fits_data = fits.open('data/SGR1900_14/SE3_8c329c0-8c337c2.evt.gz')
tte_data = fits_data[1].data['TIME']
bin_width = 0.0005
sampling_frequency = 1/bin_width
nbins = int((fits_data[1].data['TIME'][-1] - fits_data[1].data['TIME'][0])/bin_width + 1)
binned_data, _ = np.histogram(tte_data, nbins)
times = bin_width * np.arange(0, nbins)

t_start = 3293.7
t_end = 3294.0
T = t_end - t_start

idxs = np.where(np.logical_and(times > t_start, times < t_end))[0]
reduced_times = times[idxs]
reduced_binned_data = binned_data[idxs]
plt.plot(reduced_times, reduced_binned_data)
plt.show()

reduced_nbins = len(reduced_times)

def func(times,
         amplitude, t_max, sigma, skewness,
         amplitude_2, dt_2, sigma_2, skewness_2,
         amplitude_3, dt_3, sigma_3, skewness_3,
         background_rate, amplitude_qpo, phase, frequency, t_0, decay_time, **kwargs):
    T = times[-1] - times[0]
    nbin = len(times)
    norm = nbin/T
    offset = amplitude_qpo
    return burst_envelope(times=times, amplitude=amplitude, t_max=t_max, sigma=sigma, skewness=skewness)/norm + \
           burst_envelope(times=times, amplitude=amplitude_2, t_max=t_max + dt_2, sigma=sigma_2, skewness=skewness_2)/norm + \
           burst_envelope(times=times, amplitude=amplitude_3, t_max=t_max + dt_2 + dt_3, sigma=sigma_3, skewness=skewness_3)/norm + \
           background_rate/norm + \
           qpo_shot(times=times, offset=offset, amplitude=amplitude_qpo,
                    frequency=frequency, t_0=t_0, phase=phase, decay_time=decay_time)/norm


likelihood = bilby.core.likelihood.PoissonLikelihood(x=reduced_times, y=reduced_binned_data, func=func)
priors = bilby.core.prior.PriorDict(dict())
priors['background_rate'] = bilby.core.prior.LogUniform(minimum=1, maximum=1e3, name='background')

priors['amplitude'] = bilby.core.prior.LogUniform(minimum=1, maximum=1e5, name='amplitude')
priors['t_max'] = bilby.core.prior.Uniform(minimum=t_start, maximum=t_end, name='t_max')
priors['sigma'] = bilby.core.prior.LogUniform(minimum=1e-7, maximum=1, name='sigma')
priors['skewness'] = bilby.core.prior.Uniform(minimum=0, maximum=100, name='s')

# priors['amplitude_2'] = bilby.core.prior.LogUniform(minimum=1, maximum=1e5, name='amplitude_2')
# priors['dt_2'] = bilby.core.prior.Uniform(minimum=0, maximum=t_end-t_start, name='dt_2')
# priors['sigma_2'] = bilby.core.prior.LogUniform(minimum=1e-7, maximum=1, name='sigma_2')
# priors['skewness_2'] = bilby.core.prior.Uniform(minimum=0, maximum=100, name='s_2')
priors['amplitude_2'] = bilby.core.prior.DeltaFunction(peak=0, name='amplitude_2')
priors['dt_2'] = bilby.core.prior.DeltaFunction(peak=0, name='t_max_2')
priors['sigma_2'] = bilby.core.prior.DeltaFunction(peak=1, name='sigma_2')
priors['skewness_2'] = bilby.core.prior.DeltaFunction(peak=0.1, name='s_2')

# priors['amplitude_3'] = bilby.core.prior.LogUniform(minimum=1, maximum=1e5, name='amplitude_3')
# priors['dt_3'] = bilby.core.prior.Uniform(minimum=0, maximum=t_end-t_start, name='dt_3')
# priors['sigma_3'] = bilby.core.prior.LogUniform(minimum=1e-7, maximum=1, name='sigma_3')
# priors['skewness_3'] = bilby.core.prior.Uniform(minimum=0, maximum=100, name='s_3')
priors['amplitude_3'] = bilby.core.prior.DeltaFunction(peak=0, name='amplitude_3')
priors['dt_3'] = bilby.core.prior.DeltaFunction(peak=0, name='t_max_3')
priors['sigma_3'] = bilby.core.prior.DeltaFunction(peak=1, name='sigma_3')
priors['skewness_3'] = bilby.core.prior.DeltaFunction(peak=0.1, name='s_3')

priors['offset'] = bilby.core.prior.DeltaFunction(peak=0)
# priors['offset'] = bilby.core.prior.LogUniform(minimum=1, maximum=1e5, name='offset')

priors['amplitude_qpo'] = bilby.core.prior.DeltaFunction(peak=0)
priors['phase'] = bilby.core.prior.DeltaFunction(peak=0)
priors['frequency'] = bilby.core.prior.DeltaFunction(peak=0)
priors['t_0'] = bilby.core.prior.DeltaFunction(peak=0)
priors['decay_time'] = bilby.core.prior.DeltaFunction(peak=0.1)
# priors['amplitude_qpo'] = bilby.core.prior.LogUniform(minimum=1, maximum=1e4, name='amplitude_qpo')
# priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='phase')
# priors['frequency'] = bilby.core.prior.LogUniform(minimum=10/T, maximum=reduced_nbins/T, name='frequency')
# priors['t_0'] = bilby.core.prior.Uniform(minimum=t_start, maximum=t_end, name='t_0')
# priors['decay_time'] = bilby.core.prior.LogUniform(minimum=T/reduced_nbins, maximum=T, name='decay_time')
outdir = 'time_domain_real_data'
label = 'debug'
result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir, label=label,
                           sampler='dynesty', nlive=100, resume=True)

# result = bilby.core.result.Result.from_json(f'{outdir}/{label}_result.json')
result.plot_corner()
plt.plot(reduced_times, reduced_binned_data)
max_like_params = result.posterior.iloc[-1]
plt.plot(reduced_times, func(reduced_times, **max_like_params), color='r')

norm = len(reduced_times)/(reduced_times[-1] - reduced_times[0])
qpo = qpo_shot(times=reduced_times, offset=max_like_params['amplitude_qpo'], amplitude=max_like_params['amplitude_qpo'],
               frequency=max_like_params['frequency'], t_0=max_like_params['t_0'], phase=max_like_params['phase'],
               decay_time=max_like_params['decay_time']) / norm
plt.plot(reduced_times, qpo, color='g', alpha=0.2)

for i in range(10):
    params = result.posterior.iloc[np.random.randint(len(result.posterior))]
    plt.plot(reduced_times, func(reduced_times, **params), color='r', alpha=0.2)
    qpo = qpo_shot(times=reduced_times, offset=params['amplitude_qpo'], amplitude=params['amplitude_qpo'],
                   frequency=params['frequency'], t_0=params['t_0'], phase=params['phase'], decay_time=params['decay_time']) / norm
    plt.plot(reduced_times, qpo, color='g', alpha=0.2)
plt.savefig(f'{outdir}/{label}_fitted_histogram')
plt.show()
