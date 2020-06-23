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
        if isinstance(val, (float, int)):
            val = [val]
        res = np.zeros(len(val))
        for i, v in enumerate(val):
            if v >= self.spike_height:
                res[i] = self.minimum + (v - self.spike_height) / (1 - self.spike_height) \
                         * (self.maximum - self.minimum)
            else:
                res[i] = self.minimum
        if len(res) == 1:
            return res[0]
        return res

    def prob(self, val):
        return ((val >= self.minimum) & (val <= self.maximum)) * (self.spike_height + (1 - self.spike_height) / (self.maximum - self.minimum))

    def ln_prob(self, val):
        return np.log(self.prob(val))


def generic_condition_func(reference_params, amplitude):
    if isinstance(amplitude, (float, int)):
        if amplitude == 0:
            return dict(minimum=0, maximum=1e-12)
        return dict(minimum=reference_params['minimum'], maximum=reference_params['maximum'])
    else:
        res = dict(minimum=reference_params['minimum'], maximum=reference_params['maximum'])
        res['maximum'][np.where(amplitude == 0)[0]] = reference_params['minimum'][np.where(amplitude == 0)[0]] + 1e-12
        return res


def generate_qpo_prior_dict(t_start, t_end, max_burst_amplitude=1e5, max_n_bursts=1, max_qpo_amplitude=1e5,
                            max_n_qpos=1, max_background=1e4, max_frequency=1e3):
    max_sigma = t_end - t_start
    T = max_sigma
    priors = bilby.core.prior.ConditionalPriorDict(dict())
    priors['background_rate'] = bilby.core.prior.LogUniform(minimum=1, maximum=max_background, name='background')

    def condition_func_0(reference_params, amplitude_0):
        return generic_condition_func(reference_params, amplitude_0)

    def condition_func_1(reference_params, amplitude_1):
        return generic_condition_func(reference_params, amplitude_1)

    def condition_func_2(reference_params, amplitude_2):
        return generic_condition_func(reference_params, amplitude_2)

    def condition_func_3(reference_params, amplitude_3):
        return generic_condition_func(reference_params, amplitude_3)

    def condition_func_4(reference_params, amplitude_4):
        return generic_condition_func(reference_params, amplitude_4)

    def condition_func_qpo_0(reference_params, amplitude_qpo_0):
        return generic_condition_func(reference_params, amplitude_qpo_0)

    def condition_func_qpo_1(reference_params, amplitude_qpo_1):
        return generic_condition_func(reference_params, amplitude_qpo_1)

    condition_funcs = [condition_func_0, condition_func_1, condition_func_2, condition_func_3, condition_func_4]
    condition_funcs_qpo = [condition_func_qpo_0, condition_func_qpo_1]

    for i in range(max_n_bursts):
        priors[f'amplitude_{i}'] = SlabSpikePrior(minimum=0, maximum=max_burst_amplitude, spike_height=1 - 1 / (i + 1),
                                                  name=f'amplitude_{i}')
        priors[f't_max_{i}'] = bilby.core.prior.ConditionalUniform(condition_func=condition_funcs[i], minimum=t_start, maximum=t_end, name=f't_max_{i}')
        priors[f'sigma_{i}'] = bilby.core.prior.ConditionalUniform(condition_func=condition_funcs[i], minimum=1e-4, maximum=max_sigma, name=f'sigma_{i}')
        priors[f'skewness_{i}'] = bilby.core.prior.ConditionalUniform(condition_func=condition_funcs[i], minimum=0, maximum=100, name=f's_{i}')
    for i in range(max_n_bursts, 5):
        priors[f'amplitude_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f'amplitude_{i}')
        priors[f't_max_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f't_max_{i}')
        priors[f'sigma_{i}'] = bilby.core.prior.DeltaFunction(peak=1, name=f'sigma_{i}')
        priors[f'skewness_{i}'] = bilby.core.prior.DeltaFunction(peak=1, name=f's_{i}')
    for i in range(max_n_qpos):
        # priors[f'offset_{i}'] = bilby.core.prior.LogUniform(minimum=1/nbins/T, maximum=1e9, name=f'offset_{i}')
        # priors[f'phase_{i}'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name=f'phase_{i}')
        priors[f'amplitude_qpo_{i}'] = SlabSpikePrior(minimum=0, maximum=max_qpo_amplitude,
                                                      spike_height=1 - 1 / (i + 2),
                                                      name=f'amplitude_qpo_{i}')
        priors[f'frequency_{i}'] = bilby.core.prior.ConditionalUniform(condition_func=condition_funcs_qpo[i], minimum=10 / T, maximum=max_frequency, name=f'frequency_{i}')
        priors[f't_qpo_{i}'] = bilby.core.prior.ConditionalUniform(condition_func=condition_funcs_qpo[i], minimum=t_start, maximum=t_end, name=f't_qpo_{i}')
        priors[f'decay_time_{i}'] = bilby.core.prior.ConditionalUniform(condition_func=condition_funcs_qpo[i], minimum=1 / max_frequency, maximum=T, name=f'decay_time_{i}')
        priors[f'phase_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f'phase_{i}')
    for i in range(max_n_qpos, 2):
        # priors[f'offset_{i}'] = bilby.core.prior.LogUniform(minimum=1/nbins/T, maximum=1e9, name=f'offset_{i}')
        # priors[f'phase_{i}'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name=f'phase_{i}')
        priors[f'amplitude_qpo_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f'amplitude_qpo_{i}')
        priors[f'frequency_{i}'] = bilby.core.prior.DeltaFunction(peak=1, name=f'frequency_{i}')
        priors[f't_qpo_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f't_qpo_{i}')
        priors[f'decay_time_{i}'] = bilby.core.prior.DeltaFunction(peak=1, name=f'decay_time_{i}')
        priors[f'phase_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f'phase_{i}')
    return priors
