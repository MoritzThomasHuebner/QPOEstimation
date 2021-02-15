import numpy as np

import bilby

from QPOEstimation.prior.minimum import MinimumPrior


def get_kernel_prior(kernel_type, min_log_a, max_log_a, min_log_c, band_minimum, band_maximum, max_log_c=None):
    priors = dict()
    if max_log_c is None:
        max_log_c = np.log(band_maximum)

    if kernel_type == "white_noise":
        priors['kernel:log_sigma'] = bilby.core.prior.DeltaFunction(peak=-20, name='log_sigma')
    elif kernel_type == "qpo":
        if min_log_a == max_log_a:
            priors['kernel:log_a'] = bilby.core.prior.DeltaFunction(peak=max_log_a, name='log_a')
        else:
            priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='log_a')
        if min_log_c == max_log_c:
            priors['kernel:log_c'] = bilby.core.prior.DeltaFunction(peak=min_log_c, name='log_c')
        else:
            priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=max_log_c, name='log_c')

        if band_maximum == band_minimum:
            priors['kernel:log_f'] = bilby.core.prior.DeltaFunction(peak=np.log(band_minimum),
                                                                    name='log_f')
        else:
            priors['kernel:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum),
                                                              maximum=np.log(band_maximum),
                                                              name='log_f')

        priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='log_b')
        priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0,
                                                                 name='decay_constraint')
    elif kernel_type == "zeroed_qpo":
        if min_log_a == max_log_a:
            priors['kernel:log_a'] = bilby.core.prior.DeltaFunction(peak=max_log_a, name='log_a')
        else:
            priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='log_a')
        if min_log_c == max_log_c:
            priors['kernel:log_c'] = bilby.core.prior.DeltaFunction(peak=min_log_c, name='log_c')
        else:
            priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=max_log_c, name='log_c')

        if band_maximum == band_minimum:
            priors['kernel:log_f'] = bilby.core.prior.DeltaFunction(peak=np.log(band_minimum),
                                                                    name='log_f')
        else:
            priors['kernel:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum),
                                                              maximum=np.log(band_maximum),
                                                              name='log_f')
        priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0,
                                                                 name='decay_constraint')
    elif kernel_type == "red_noise":
        if min_log_a == max_log_a:
            priors['kernel:log_a'] = bilby.core.prior.DeltaFunction(peak=max_log_a, name='log_a')
        else:
            priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='log_a')
        if min_log_c == max_log_c:
            priors['kernel:log_c'] = bilby.core.prior.DeltaFunction(peak=min_log_c, name='log_c')
        else:
            priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=max_log_c, name='log_c')
    elif kernel_type == "mixed":
        priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a,
                                                                   name='terms[0]:log_a')
        priors['kernel:terms[0]:log_b'] = bilby.core.prior.DeltaFunction(peak=-15, name='terms[0]:log_b')
        priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=max_log_c,
                                                                   name='terms[0]:log_c')
        priors['kernel:terms[0]:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum),
                                                                   maximum=np.log(band_maximum),
                                                                   name='terms[0]:log_f')
        priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a,
                                                                   name='terms[1]:log_a')
        priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=max_log_c,
                                                                   name='terms[1]:log_c')
        priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0,
                                                                 name='decay_constraint')
    elif kernel_type == "zeroed_mixed":
        priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a,
                                                                   name='terms[0]:log_a')
        priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=max_log_c,
                                                                   name='terms[0]:log_c')
        priors['kernel:terms[0]:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum),
                                                                   maximum=np.log(band_maximum),
                                                                   name='terms[0]:log_f')
        priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a,
                                                                   name='terms[1]:log_a')
        priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=max_log_c,
                                                                   name='terms[1]:log_c')
        priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0,
                                                                 name='decay_constraint')
    else:
        raise ValueError('Recovery mode not defined')
    return priors

def get_polynomial_prior(polynomial_max=10):
    priors = dict()
    if polynomial_max == 0:
        priors['mean:a0'] = 0
        priors['mean:a1'] = 0
        priors['mean:a2'] = 0
        priors['mean:a3'] = 0
        priors['mean:a4'] = 0
    else:
        priors['mean:a0'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a0')
        priors['mean:a1'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a1')
        priors['mean:a2'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a2')
        priors['mean:a3'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a3')
        priors['mean:a4'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a4')
    return priors


def get_window_priors(times):
    priors = dict()
    priors['window_minimum'] = bilby.core.prior.Beta(minimum=times[0], maximum=times[-1], alpha=1, beta=2, name='window_minimum')
    priors['window_maximum'] = MinimumPrior(minimum=times[0], maximum=times[-1], order=1, reference_name='window_minimum', name='window_maximum', minimum_spacing=0.1)
    # priors['window_maximum'] = bilby.core.prior.Uniform(minimum=times[0], maximum=times[-1], name='window_maximum')
    # priors['window_minimum'] = bilby.core.prior.Uniform(minimum=times[0], maximum=times[0]+0.3, name='window_minimum')
    # priors['window_size'] = bilby.core.prior.Uniform(minimum=0.3, maximum=0.7, name='window_size')
    # priors['window_maximum'] = bilby.core.prior.Constraint(minimum=-1000, maximum=times[-1], name='window_maximum')
    # priors['window_minimum'] = bilby.core.prior.Uniform(minimum=times[0], maximum=times[-1], name='window_minimum')
    # priors['window_size'] = bilby.core.prior.Uniform(minimum=0.1, maximum=times[-1] - times[0], name='window_size')
    # priors['window_maximum'] = bilby.core.prior.Constraint(minimum=-1000, maximum=times[-1], name='window_maximum')
    return priors


def decay_constrain_conversion_function(sample):
    out_sample = sample.copy()
    if 'kernel:log_c' in sample.keys():
        out_sample['decay_constraint'] = out_sample['kernel:log_c'] - out_sample['kernel:log_f']
    else:
        out_sample['decay_constraint'] = out_sample['kernel:terms[0]:log_c'] - out_sample['kernel:terms[0]:log_f']
    return out_sample
