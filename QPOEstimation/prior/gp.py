import numpy as np
from copy import deepcopy

import bilby

from QPOEstimation.prior.minimum import MinimumPrior


def get_kernel_prior(kernel_type, min_log_a, max_log_a, min_log_c, band_minimum,
                     band_maximum, max_log_c=np.nan, jitter_term=False, **kwargs):
    if max_log_c is None or np.isnan(max_log_c):
        max_log_c = np.log(band_maximum)

    if kernel_type == "white_noise":
        priors = get_white_noise_prior()
    elif kernel_type == "qpo":
        priors = get_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c)
    elif kernel_type == "pure_qpo":
        priors = get_pure_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c)
    elif kernel_type == "red_noise":
        priors = get_red_noise_prior(max_log_a, max_log_c, min_log_a, min_log_c)
    elif kernel_type == "double_red_noise":
        priors = get_double_red_noise_prior(max_log_a, max_log_c, min_log_a, min_log_c)
    elif kernel_type == "general_qpo":
        priors = get_general_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c)
    elif kernel_type == "double_qpo":
        priors = get_double_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c)
    elif kernel_type == "fourier_series":
        priors = get_fourier_series_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c)
    else:
        raise ValueError('Recovery mode not defined')
    if jitter_term:
        priors = _add_jitter_term(priors)
    return priors


def _add_jitter_term(priors):
    new_priors = deepcopy(priors)
    for k, v in priors.items():
        if ':' in k:
            if '[' not in k:
                new_key = k.replace(':', ':terms[0]:')
                new_priors[new_key] = new_priors[k]
                new_priors[new_key].name = f"terms[0]:{new_priors[new_key].name}"
                del new_priors[k]
    n_terms = 0
    for k in new_priors.keys():
        for i in range(10):
            if f"[{i}]" in k and n_terms < i + 1:
                n_terms = i + 1
    new_priors[f'kernel:terms[{n_terms}]:log_sigma'] = bilby.core.prior.Uniform(minimum=-10, maximum=5, name=f'terms[{n_terms}]:log_sigma')
    return new_priors


def get_white_noise_prior():
    priors = bilby.prior.PriorDict()
    priors['kernel:log_sigma'] = bilby.core.prior.DeltaFunction(peak=-20, name='log_sigma')
    return priors


def get_general_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c):
    min_log_f = np.log(band_minimum)
    max_log_f = np.log(band_maximum)

    priors = bilby.prior.PriorDict()
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[0]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_c, maximum=max_log_c, label='terms[0]:log_c')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_f, maximum=max_log_f, label='terms[0]:log_f')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[1]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_c, maximum=max_log_c, label='terms[1]:log_c')
    priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name='decay_constraint')
    priors.conversion_function = decay_constrain_conversion_function
    return priors


def get_red_noise_prior(max_log_a, max_log_c, min_log_a, min_log_c):
    priors = bilby.prior.PriorDict()
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_c, maximum=max_log_c, label='log_c')
    return priors


def get_double_red_noise_prior(max_log_a, max_log_c, min_log_a, min_log_c):
    priors = bilby.prior.ConditionalPriorDict()
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[0]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_c, maximum=max_log_c, label='terms[0]:log_c')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[1]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_c, maximum=max_log_c, label='terms[1]:log_c')

    priors['kernel:terms[0]:log_a'] = \
        bilby.core.prior.Beta(
            minimum=min_log_a, maximum=max_log_a, alpha=1, beta=2,
            name='kernel:terms[0]:log_a', boundary='reflective')
    priors['kernel:terms[1]:log_a'] = MinimumPrior(
        minimum=min_log_a, maximum=max_log_a, order=1, reference_name='kernel:terms[0]:log_a',
        name='kernel:terms[1]:log_a', minimum_spacing=0.0, boundary='reflective')
    return priors


def get_double_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c):
    min_log_f = np.log(band_minimum)
    max_log_f = np.log(band_maximum)

    priors = bilby.prior.ConditionalPriorDict()
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[0]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_c, maximum=max_log_c, label='terms[0]:log_c')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[1]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_c, maximum=max_log_c, label='terms[1]:log_c')
    priors['kernel:terms[0]:log_f'] = \
        bilby.core.prior.Beta(
            minimum=min_log_f, maximum=max_log_f, alpha=1, beta=2,
            name='kernel:terms[0]:log_f', boundary='reflective')
    priors['kernel:terms[1]:log_f'] = MinimumPrior(
        minimum=min_log_f, maximum=max_log_f, order=1, reference_name='kernel:terms[0]:log_f',
        name='kernel:terms[1]:log_f', minimum_spacing=0.0, boundary='reflective')

    priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name='decay_constraint')
    priors['decay_constraint_2'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name='decay_constraint_2')
    priors.conversion_function = decay_constrain_conversion_function
    return priors


def get_fourier_series_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c):
    min_log_f = np.log(band_minimum)
    max_log_f = np.log(band_maximum)

    priors = bilby.prior.ConditionalPriorDict()
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[0]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[1]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[2]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[3]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[4]:log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='terms[5]:log_a')

    _add_individual_kernel_prior(priors=priors, minimum=min_log_c, maximum=max_log_c, label='terms[0]:log_c')

    _add_individual_kernel_prior(priors=priors, minimum=min_log_f, maximum=max_log_f, label='terms[0]:log_f')

    def c_condition_func(reference_params, **kwargs):
        return dict(peak=kwargs["kernel:terms[0]:log_c"])


    def f_condition_func_1(reference_params, **kwargs):
        return dict(peak=kwargs["kernel:terms[0]:log_f"] + np.log(2))

    def f_condition_func_2(reference_params, **kwargs):
        return dict(peak=kwargs["kernel:terms[0]:log_f"] + np.log(3))

    def f_condition_func_3(reference_params, **kwargs):
        return dict(peak=kwargs["kernel:terms[0]:log_f"] + np.log(4))

    def f_condition_func_4(reference_params, **kwargs):
        return dict(peak=kwargs["kernel:terms[0]:log_f"] + np.log(5))

    def f_condition_func_5(reference_params, **kwargs):
        return dict(peak=kwargs["kernel:terms[0]:log_f"] + np.log(6))

    priors['kernel:terms[1]:log_c'] = bilby.core.prior.ConditionalDeltaFunction(peak=-3, condition_func=c_condition_func, name="terms[1]:log_c")#, reference_name='kernel:terms[0]:log_c')
    priors['kernel:terms[2]:log_c'] = bilby.core.prior.ConditionalDeltaFunction(peak=-3, condition_func=c_condition_func, name="terms[2]:log_c")#, reference_name='kernel:terms[0]:log_c')
    priors['kernel:terms[3]:log_c'] = bilby.core.prior.ConditionalDeltaFunction(peak=-3, condition_func=c_condition_func, name="terms[3]:log_c")#, reference_name='kernel:terms[0]:log_c')
    priors['kernel:terms[4]:log_c'] = bilby.core.prior.ConditionalDeltaFunction(peak=-3, condition_func=c_condition_func, name="terms[4]:log_c")#, reference_name='kernel:terms[0]:log_c')
    priors['kernel:terms[5]:log_c'] = bilby.core.prior.ConditionalDeltaFunction(peak=-3, condition_func=c_condition_func, name="terms[5]:log_c")#, reference_name='kernel:terms[0]:log_c')

    priors['kernel:terms[1]:log_c']._required_variables = ['kernel:terms[0]:log_c']
    priors['kernel:terms[2]:log_c']._required_variables = ['kernel:terms[0]:log_c']
    priors['kernel:terms[3]:log_c']._required_variables = ['kernel:terms[0]:log_c']
    priors['kernel:terms[4]:log_c']._required_variables = ['kernel:terms[0]:log_c']
    priors['kernel:terms[5]:log_c']._required_variables = ['kernel:terms[0]:log_c']

    priors['kernel:terms[1]:log_f'] = bilby.core.prior.ConditionalDeltaFunction(peak=-3, condition_func=f_condition_func_1, name="terms[1]:log_f")#, reference_name='kernel:terms[0]:log_f')
    priors['kernel:terms[2]:log_f'] = bilby.core.prior.ConditionalDeltaFunction(peak=-3, condition_func=f_condition_func_2, name="terms[2]:log_f")#, reference_name='kernel:terms[0]:log_f')
    priors['kernel:terms[3]:log_f'] = bilby.core.prior.ConditionalDeltaFunction(peak=-3, condition_func=f_condition_func_3, name="terms[3]:log_f")#, reference_name='kernel:terms[0]:log_f')
    priors['kernel:terms[4]:log_f'] = bilby.core.prior.ConditionalDeltaFunction(peak=-3, condition_func=f_condition_func_4, name="terms[4]:log_f")#, reference_name='kernel:terms[0]:log_f')
    priors['kernel:terms[5]:log_f'] = bilby.core.prior.ConditionalDeltaFunction(peak=-3, condition_func=f_condition_func_5, name="terms[5]:log_f")#, reference_name='kernel:terms[0]:log_f')

    priors['kernel:terms[1]:log_f']._required_variables = ['kernel:terms[0]:log_f']
    priors['kernel:terms[2]:log_f']._required_variables = ['kernel:terms[0]:log_f']
    priors['kernel:terms[3]:log_f']._required_variables = ['kernel:terms[0]:log_f']
    priors['kernel:terms[4]:log_f']._required_variables = ['kernel:terms[0]:log_f']
    priors['kernel:terms[5]:log_f']._required_variables = ['kernel:terms[0]:log_f']

    priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name='decay_constraint')
    priors.conversion_function = decay_constrain_conversion_function
    priors._resolve_conditions()
    return priors


def get_pure_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c):
    min_log_f = np.log(band_minimum)
    max_log_f = np.log(band_maximum)

    priors = bilby.prior.PriorDict()
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_c, maximum=max_log_c, label='log_c')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_f, maximum=max_log_f, label='log_f')
    priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name='decay_constraint')
    priors.conversion_function = decay_constrain_conversion_function
    return priors


def get_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c):
    min_log_f = np.log(band_minimum)
    max_log_f = np.log(band_maximum)

    priors = bilby.prior.PriorDict()
    priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='log_b')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label='log_a')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_c, maximum=max_log_c, label='log_c')
    _add_individual_kernel_prior(priors=priors, minimum=min_log_f, maximum=max_log_f, label='log_f')
    priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name='decay_constraint')
    priors.conversion_function = decay_constrain_conversion_function
    return priors


def _add_individual_kernel_prior(priors, minimum, maximum, label):
    if minimum == maximum:
        priors[f'kernel:{label}'] = bilby.core.prior.DeltaFunction(peak=maximum, name=label)
    else:
        priors[f'kernel:{label}'] = bilby.core.prior.Uniform(minimum=minimum, maximum=maximum, name=label)


def get_window_priors(times, likelihood_model='gaussian_process_windowed', **kwargs):
    if likelihood_model == 'gaussian_process_windowed':
        priors = bilby.core.prior.ConditionalPriorDict()
        priors['window_minimum'] = bilby.core.prior.Beta(minimum=times[0], maximum=times[-1], alpha=1, beta=2,
                                                         name='window_minimum', boundary='reflective')
        priors['window_maximum'] = MinimumPrior(minimum=times[0], maximum=times[-1], order=1,
                                                reference_name='window_minimum', name='window_maximum',
                                                minimum_spacing=0.1, boundary='reflective')
        return priors
    else:
        return bilby.prior.PriorDict()


def decay_constrain_conversion_function(sample):
    out_sample = sample.copy()
    if 'kernel:log_f' in sample.keys():
        out_sample['decay_constraint'] = out_sample['kernel:log_c'] - out_sample['kernel:log_f']
    elif "kernel:terms[0]:log_f" in sample.keys():
        out_sample['decay_constraint'] = out_sample['kernel:terms[0]:log_c'] - out_sample['kernel:terms[0]:log_f']
    return out_sample


def decay_constrain_conversion_function_2(sample):
    out_sample = sample.copy()
    out_sample['decay_constraint'] = out_sample['kernel:terms[0]:log_c'] - out_sample['kernel:terms[0]:log_f']
    out_sample['decay_constraint_2'] = out_sample['kernel:terms[1]:log_c'] - out_sample['kernel:terms[1]:log_f']
    return out_sample
