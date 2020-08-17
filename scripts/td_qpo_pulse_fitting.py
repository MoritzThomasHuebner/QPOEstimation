import matplotlib

from QPOEstimation.model.series import burst_qpo_model_norm

matplotlib.use("Qt5Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
from QPOEstimation.prior import *

fits_data = fits.open('data/SGR1900_14/SE3_8c329c0-8c337c2.evt.gz')
tte_data = fits_data[1].data['TIME']
bin_width = 0.0005
sampling_frequency = 1 / bin_width
nbins = int((fits_data[1].data['TIME'][-1] - fits_data[1].data['TIME'][0]) / bin_width + 1)
binned_data, _ = np.histogram(tte_data, nbins)
times = bin_width * np.arange(0, nbins)

t_start = 3293.5
t_end = 3294.5
T = t_end - t_start

idxs = np.where(np.logical_and(times > t_start, times < t_end))[0]
reduced_times = times[idxs]
reduced_binned_data = binned_data[idxs]
plt.plot(reduced_times, reduced_binned_data)
plt.show()

reduced_nbins = len(reduced_times)

max_n_bursts = 2
max_n_qpos = 0
max_burst_amplitude = 1e6
max_background = 1e3
max_qpo_amplitude = 10000
priors = generate_qpo_prior_dict(t_start=t_start, t_end=t_end, max_burst_amplitude=max_burst_amplitude,
                                 max_n_bursts=max_n_bursts,
                                 max_qpo_amplitude=max_qpo_amplitude, max_n_qpos=max_n_qpos,
                                 max_background=max_background, max_frequency=nbins / T)

priors[f'amplitude_1'].spike_height = 0.01
# priors[f'amplitude_2'].spike_height = 0.0

# priors[f'amplitude_0'] = bilby.core.prior.Uniform(minimum=10000, maximum=max_burst_amplitude, name=f'amplitude_0')
# priors[f't_max_0'] = bilby.core.prior.Uniform(minimum=t_start, maximum=t_end, name=f't_max_0')
# priors[f'sigma_0'] = bilby.core.prior.Uniform(minimum=0, maximum=max_background, name=f'sigma_0')
# priors[f'skewness_0'] = bilby.core.prior.Uniform(minimum=0, maximum=500, name=f's_0')
#
# priors[f'amplitude_1'] = SlabSpikePrior(minimum=0, maximum=max_burst_amplitude, spike_height=0.5, name=f'amplitude_1')
# priors[f't_max_1'] = bilby.core.prior.Uniform(minimum=t_start, maximum=t_end, name=f't_max_1')
# priors[f'sigma_1'] = bilby.core.prior.Uniform(minimum=0, maximum=max_background, name=f'sigma_1')
# priors[f'skewness_1'] = bilby.core.prior.Uniform(minimum=0, maximum=500, name=f's_1')
# priors['background_rate'] = bilby.core.prior.Uniform(minimum=0, maximum=max_background, name='background')
#
# def condition_func_freq(reference_params, decay_time_0):
#     max_freq = 1/decay_time_0
#     return dict(minimum=reference_params['minimum'], maximum=max_freq)
#
# priors[f'amplitude_qpo_0'] = SlabSpikePrior(minimum=0, maximum=max_qpo_amplitude, spike_height=0.95, name=f'amplitude_qpo_0')
# priors[f'frequency_0'] = bilby.core.prior.ConditionalUniform( condition_func=condition_func_freq, minimum=10 / T, maximum=1000, name=f'frequency_0')
# priors[f't_qpo_0'] = bilby.core.prior.Uniform(minimum=t_start, maximum=t_end, name=f't_qpo_0')
# priors[f'decay_time_0'] = bilby.core.prior.Uniform(minimum=1 / 100, maximum=T, name=f'decay_time_0')
# priors[f'phase_0'] = bilby.core.prior.DeltaFunction(peak=0, name=f'phase_0')

likelihood = bilby.core.likelihood.PoissonLikelihood(x=reduced_times, y=reduced_binned_data, func=burst_qpo_model_norm)

outdir = 'time_domain_real_data'
label = 'slab_spike_long_segment'
result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir, label=label,
                           sampler='dynesty', nlive=500, sample='rslice', resume=True, nact=10)

for i in range(max_n_bursts):
    parameters = [f'amplitude_{i}', f't_max_{i}', f'sigma_{i}', f'skewness_{i}']
    if i == 0:
        parameters.append('background_rate')
    result.plot_corner(parameters=parameters, filename=f'{outdir}/{label}_corner_{i}')
for i in range(max_n_qpos):
    result.plot_corner(
        parameters=[f'amplitude_qpo_{i}', f'frequency_{i}', f't_qpo_{i}', f'decay_time_{i}', f'phase_{i}'],
        filename=f'{outdir}/{label}_corner_qpo_{i}')

print(len(result.posterior))

max_like_params = result.posterior.iloc[-1]
plt.plot(reduced_times, reduced_binned_data)
plt.plot(reduced_times, burst_qpo_model_norm(reduced_times, **max_like_params), color='r')

norm = len(reduced_times) / (reduced_times[-1] - reduced_times[0])
# qpo = qpo_shot(times=reduced_times, offset=max_like_params['amplitude_qpo'], amplitude=max_like_params['amplitude_qpo'],
#                frequency=max_like_params['frequency'], t_qpo=max_like_params['t_0'], phase=max_like_params['phase'],
#                decay_time=max_like_params['decay_time']) / norm
# plt.plot(reduced_times, qpo, color='g', alpha=0.2)

for i in range(100):
    params = result.posterior.iloc[np.random.randint(len(result.posterior))]
    plt.plot(reduced_times, burst_qpo_model_norm(reduced_times, **params), color='r', alpha=0.2)
    # qpo = qpo_shot(times=reduced_times, offset=params['amplitude_qpo'], amplitude=params['amplitude_qpo'],
    #                frequency=params['frequency'], t_qpo=params['t_0'], phase=params['phase'], decay_time=params['decay_time']) / norm
    # plt.plot(reduced_times, qpo, color='g', alpha=0.2)
plt.savefig(f'{outdir}/{label}_fitted_histogram')
plt.show()