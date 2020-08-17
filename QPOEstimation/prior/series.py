import bilby
import numpy as np

from QPOEstimation.prior.slabspike import SlabSpikePrior, ConditionalSlabSpikePrior


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