import numpy as np

import bilby

from QPOEstimation.prior.minimum import MinimumPrior


def get_kernel_prior(kernel_type, min_log_a, max_log_a, min_log_c, band_minimum,
                     band_maximum, max_log_c=np.nan, **kwargs):
    if np.isnan(max_log_c):
        max_log_c = np.log(band_maximum)

    if kernel_type == "white_noise":
        priors = get_white_noise_prior()
    elif kernel_type == "qpo":
        priors = get_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c)
    elif kernel_type == "pure_qpo":
        priors = get_pure_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c)
    elif kernel_type == "red_noise":
        priors = get_red_noise_prior(max_log_a, max_log_c, min_log_a, min_log_c)
    elif kernel_type == "general_qpo":
        priors = get_general_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c)
    else:
        raise ValueError('Recovery mode not defined')
    return priors


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
