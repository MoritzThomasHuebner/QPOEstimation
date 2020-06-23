import bilby
import numpy as np


def get_red_noise_prior():
    prior = bilby.core.prior.PriorDict()
    prior['alpha'] = bilby.core.prior.Uniform(0, 10, name='alpha')
    prior['beta'] = bilby.core.prior.LogUniform(1e-6, 10, name='beta')
    prior['sigma'] = bilby.core.prior.LogUniform(1e-5, 1, name='sigma')
    return prior


def get_qpo_prior(frequencies=None):
    if frequencies is None:
        df = 1
        max_frequency = 512
    else:
        df = frequencies[1] - frequencies[0]
        max_frequency = frequencies[-1]
    prior = bilby.core.prior.PriorDict()
    prior['amplitude'] = bilby.core.prior.LogUniform(1e-4, 100, name='amplitude')
    prior['width'] = bilby.core.prior.Uniform(5*df/np.pi, 100, name='width') # 5 time FWHM as a minimum
    prior['central_frequency'] = bilby.core.prior.Uniform(1.0, max_frequency, name='central_frequency')
    # prior['offset'] = bilby.core.prior.Uniform(0, 1, name='offset')
    prior['offset'] = bilby.core.prior.DeltaFunction(0, name='offset')
    return prior


def get_broken_power_law_prior():
    prior = bilby.core.prior.PriorDict()
    prior['alpha_1'] = bilby.core.prior.Uniform(0, 10, name='alpha_1')
    prior['alpha_2'] = bilby.core.prior.Uniform(0, 10, name='alpha_2')
    prior['delta'] = bilby.core.prior.LogUniform(1e-4, 100, name='delta')
    prior['rho'] = bilby.core.prior.DeltaFunction(peak=-1)
    prior['beta'] = bilby.core.prior.LogUniform(0.01, 10, name='beta')
    prior['sigma'] = bilby.core.prior.LogUniform(1e-4, 1, name='sigma')
    return prior


def get_full_prior(noise_model='red_noise', frequencies=None):
    if noise_model == 'broken_power_law':
        noise_prior = get_broken_power_law_prior()
    else:
        noise_prior = get_red_noise_prior()
    prior = get_qpo_prior(frequencies=frequencies)
    for key, value in noise_prior.items():
        prior[key] = value
    return prior


class SlabSpikePrior(bilby.core.prior.Prior):

    def __init__(self, name=None, latex_label=None, unit=None, minimum=0.,
                 maximum=1., spike_height=0.5, check_range_nonzero=True, boundary=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit, minimum=minimum,
                         maximum=maximum, check_range_nonzero=check_range_nonzero, boundary=boundary)
        self.spike_loc = minimum
        self.spike_height = spike_height

    def rescale(self, val):
        self.test_valid_for_rescaling(val)
        if val >= self.spike_height:
            return self.minimum + (val - self.spike_height) / (1 - self.spike_height) * (self.maximum - self.minimum)
        else:
            return self.minimum

    def prob(self, val):
        return ((val >= self.minimum) & (val <= self.maximum)) * (self.spike_height + (1 - self.spike_height) / (self.maximum - self.minimum))

    def ln_prob(self, val):
        return np.log(self.prob(val))


def generate_qpo_prior_dict(t_start, t_end, max_burst_amplitude=1e5, max_n_bursts=1, max_qpo_amplitude=1e5,
                            max_n_qpos=1, max_background=1e4, max_frequency=1e3, ):
    max_sigma = t_end - t_start
    T = max_sigma
    priors = bilby.core.prior.PriorDict(dict())
    priors['background_rate'] = bilby.core.prior.LogUniform(minimum=1, maximum=max_background, name='background')
    for i in range(max_n_bursts):
        priors[f'amplitude_{i}'] = SlabSpikePrior(minimum=0, maximum=max_burst_amplitude, spike_height=1 - 1/(i + i), name=f'amplitude_{i}')
        priors[f't_max_{i}'] = bilby.core.prior.Uniform(minimum=t_start, maximum=t_end, name=f't_max_{i}')
        priors[f'sigma_{i}'] = bilby.core.prior.LogUniform(minimum=1e-4, maximum=max_sigma, name=f'sigma_{i}')
        priors[f'skewness_{i}'] = bilby.core.prior.Uniform(minimum=0, maximum=100, name=f's_{i}')
    for i in range(max_n_qpos):
        priors['amplitude_qpo'] = SlabSpikePrior(minimum=0, maximum=max_qpo_amplitude, name='amplitude_qpo')
        # priors['offset'] = bilby.core.prior.LogUniform(minimum=1/nbins/T, maximum=1e9, name='offset')
        # priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='phase')
        priors['frequency'] = bilby.core.prior.LogUniform(minimum=10 / T, maximum=max_frequency, name='frequency')
        priors['t_0'] = bilby.core.prior.Uniform(minimum=t_start, maximum=t_end, name='t_0')
        priors['decay_time'] = bilby.core.prior.LogUniform(minimum=1/max_frequency, maximum=T, name='decay_time')
        priors['phase'] = bilby.core.prior.DeltaFunction(peak=0, name='phase')