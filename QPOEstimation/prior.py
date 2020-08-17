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
    prior['width'] = bilby.core.prior.Uniform(df/np.pi, 100, name='width')
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
                 maximum=1., spike_height=0.5, spike_loc=None, check_range_nonzero=True, boundary=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit, minimum=minimum,
                         maximum=maximum, check_range_nonzero=check_range_nonzero, boundary=boundary)
        if spike_loc is None:
            self.spike_loc = minimum
        else:
            self.spike_loc = spike_loc
        self.spike_height = spike_height

    @property
    def segment_length(self):
        return self.maximum - self.minimum

    def rescale(self, val):
        val = np.atleast_1d(val)
        res = np.zeros(len(val))
        non_spike_frac = 1 - self.spike_height
        frac_below_spike = (self.spike_loc - self.minimum)/self.segment_length * non_spike_frac
        spike_start = frac_below_spike
        lower_indices = np.where(val < spike_start)
        intermediate_indices = np.where(np.logical_and(val >= spike_start, val <= spike_start + self.spike_height))
        higher_indices = np.where(val > spike_start + self.spike_height)
        res[lower_indices] = val[lower_indices] * self.segment_length / non_spike_frac + self.minimum
        res[intermediate_indices] = spike_start * self.segment_length / non_spike_frac + self.minimum
        res[higher_indices] = (val[higher_indices] - self.spike_height) * self.segment_length / non_spike_frac + self.minimum
        return res

    def prob(self, val):
        return ((val >= self.minimum) & (val <= self.maximum)) * (self.spike_height + (1 - self.spike_height) / (self.maximum - self.minimum))

    def ln_prob(self, val):
        return np.log(self.prob(val))


ConditionalSlabSpikePrior = bilby.core.prior.conditional_prior_factory(SlabSpikePrior)


def generic_condition_func(reference_params, amplitude):
    if isinstance(amplitude, (float, int)):
        if amplitude == 0:
            return dict(minimum=0, maximum=1e-12)
        return dict(minimum=reference_params['minimum'], maximum=reference_params['maximum'])
    else:
        return dict(minimum=reference_params['minimum'], maximum=reference_params['maximum'])


def generic_condition_func_amplitude_qpo(reference_params, amplitude_spike):
    reference_params_copy = reference_params.copy()
    amplitude_spike = np.atleast_1d(amplitude_spike)
    spike_heights = np.atleast_1d(reference_params_copy["spike_height"])
    spike_heights[np.where(amplitude_spike == 0)] = 1
    reference_params_copy["spike_height"] = spike_heights
    reference_params_copy["maximum"] = amplitude_spike
    return reference_params_copy


def generic_t_qpo_condition_func(reference_params, t_spike, decay_time, skewness):
    reference_params_copy = reference_params.copy()
    reference_params_copy['minimum'] = t_spike
    reference_params_copy['maximum'] = t_spike + 2*decay_time*skewness
    return reference_params_copy


def generic_f_qpo_condition_func(reference_params, decay_time, skewness):
    reference_params_copy = reference_params.copy()
    reference_params_copy['minimum'] = 1/(decay_time * skewness)
    return reference_params_copy


def amplitude_qpo_condition_func_0(reference_params, amplitude_spike_0):
    return generic_condition_func_amplitude_qpo(reference_params, amplitude_spike_0)
def amplitude_qpo_condition_func_1(reference_params, amplitude_spike_1):
    return generic_condition_func_amplitude_qpo(reference_params, amplitude_spike_1)
def amplitude_qpo_condition_func_2(reference_params, amplitude_spike_2):
    return generic_condition_func_amplitude_qpo(reference_params, amplitude_spike_2)
def amplitude_qpo_condition_func_3(reference_params, amplitude_spike_3):
    return generic_condition_func_amplitude_qpo(reference_params, amplitude_spike_3)
def amplitude_qpo_condition_func_4(reference_params, amplitude_spike_4):
    return generic_condition_func_amplitude_qpo(reference_params, amplitude_spike_4)


def t_qpo_condition_func_0(reference_params, t_spike_0, decay_time_0, skewness_0):
    return generic_t_qpo_condition_func(reference_params, t_spike_0, decay_time_0, skewness_0)
def t_qpo_condition_func_1(reference_params, t_spike_1, decay_time_1, skewness_1):
    return generic_t_qpo_condition_func(reference_params, t_spike_1, decay_time_1, skewness_1)
def t_qpo_condition_func_2(reference_params, t_spike_2, decay_time_2, skewness_2):
    return generic_t_qpo_condition_func(reference_params, t_spike_2, decay_time_2, skewness_2)
def t_qpo_condition_func_3(reference_params, t_spike_3, decay_time_3, skewness_3):
    return generic_t_qpo_condition_func(reference_params, t_spike_3, decay_time_3, skewness_3)
def t_qpo_condition_func_4(reference_params, t_spike_4, decay_time_4, skewness_4):
    return generic_t_qpo_condition_func(reference_params, t_spike_4, decay_time_4, skewness_4)


def f_qpo_condition_func_0(reference_params, decay_time_0, skewness_0):
    return generic_f_qpo_condition_func(reference_params, decay_time_0, skewness_0)
def f_qpo_condition_func_1(reference_params, decay_time_1, skewness_1):
    return generic_f_qpo_condition_func(reference_params, decay_time_1, skewness_1)
def f_qpo_condition_func_2(reference_params, decay_time_2, skewness_2):
    return generic_f_qpo_condition_func(reference_params, decay_time_2, skewness_2)
def f_qpo_condition_func_3(reference_params, decay_time_3, skewness_3):
    return generic_f_qpo_condition_func(reference_params, decay_time_3, skewness_3)
def f_qpo_condition_func_4(reference_params, decay_time_4, skewness_4):
    return generic_f_qpo_condition_func(reference_params, decay_time_4, skewness_4)


amplitude_qpo_condition_funcs = [amplitude_qpo_condition_func_0, amplitude_qpo_condition_func_1,
                                 amplitude_qpo_condition_func_2, amplitude_qpo_condition_func_3,
                                 amplitude_qpo_condition_func_4]
t_qpo_condition_funcs = [t_qpo_condition_func_0, t_qpo_condition_func_1,
                         t_qpo_condition_func_2, t_qpo_condition_func_3,
                         t_qpo_condition_func_4]
f_qpo_condition_funcs = [f_qpo_condition_func_0, f_qpo_condition_func_1,
                         f_qpo_condition_func_2, f_qpo_condition_func_3,
                         f_qpo_condition_func_4]


def generate_qpo_prior_dict(t_start, t_end, max_burst_amplitude=5e5, max_n_bursts=1, max_background=1e5,
                            spike_height=0.1, spike_height_qpo=0.1):
    max_decay_time = t_end - t_start
    priors = bilby.core.prior.ConditionalPriorDict(dict())
    priors['background_rate'] = bilby.core.prior.LogUniform(minimum=1e-8, maximum=max_background, name='background')

    for i in range(max_n_bursts):
        priors[f'amplitude_spike_{i}'] = SlabSpikePrior(minimum=0, maximum=max_burst_amplitude,
                                                        spike_height=spike_height, name=f'amplitude_spike_{i}')
        priors[f't_spike_{i}'] = bilby.core.prior.Uniform(minimum=t_start, maximum=t_end, name=f't_spike_{i}')
        priors[f'amplitude_qpo_{i}'] = ConditionalSlabSpikePrior(
            condition_func=amplitude_qpo_condition_funcs[i], minimum=0, maximum=max_burst_amplitude,
            spike_height=spike_height_qpo, name=f'amplitude_qpo_{i}')
        priors[f't_qpo_{i}'] = bilby.core.prior.ConditionalUniform(
            condition_func=t_qpo_condition_funcs[i], minimum=t_start, maximum=t_end, name=f't_qpo_{i}')
        priors[f'phase_{i}'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name=f'phase_{i}')
        priors[f'decay_time_{i}'] = bilby.core.prior.LogUniform(minimum=1e-5, maximum=max_decay_time, name=f'tau_{i}')
        priors[f'skewness_{i}'] = bilby.core.prior.LogUniform(minimum=1e-5, maximum=10000, name=f's_{i}')
        priors[f'f_qpo_{i}'] = bilby.core.prior.ConditionalLogUniform(condition_func=f_qpo_condition_funcs[i],
                                                                      minimum=1/(t_end-t_start), maximum=1e3, name=f'f_qpo_{i}')
    for i in range(max_n_bursts, 5):
        priors[f'amplitude_spike_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f'amplitude_spike_{i}')
        priors[f't_spike_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f't_spike_{i}')
        priors[f'amplitude_qpo_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f'amplitude_qpo_{i}')
        priors[f't_qpo_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f't_qpo_{i}')
        priors[f'phase_{i}'] = bilby.core.prior.DeltaFunction(peak=0, name=f'phase_{i}')
        priors[f'decay_time_{i}'] = bilby.core.prior.DeltaFunction(peak=1, name=f'tau_{i}')
        priors[f'skewness_{i}'] = bilby.core.prior.DeltaFunction(peak=1, name=f's_{i}')
        priors[f'f_qpo_{i}'] = bilby.core.prior.DeltaFunction(peak=1, name=f't_qpo_{i}')
    return priors
