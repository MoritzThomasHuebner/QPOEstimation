import numpy as np

import bilby

from . import minimum, psd, gp, mean


def get_priors(**kwargs):
    """ Catch all function"""

    segment_length = kwargs['times'][-1] - kwargs['times'][0]
    sampling_frequency = 1 / (kwargs['times'][1] - kwargs['times'][0])
    if kwargs['min_log_c'] is None:
        if kwargs['kernel_type'] == 'red_noise':
            kwargs['min_log_c'] = np.log(1 / segment_length)
        else:
            kwargs['min_log_c'] = -10
    if kwargs['max_log_c'] is None:
        if kwargs['kernel_type'] == 'red_noise':
            kwargs['max_log_c'] = np.log(sampling_frequency)
        else:
            kwargs['max_log_c'] = np.log(kwargs['band_maximum'])

    priors = bilby.core.prior.ConditionalPriorDict()
    mean_priors = mean.get_mean_prior(**kwargs)
    kernel_priors = gp.get_kernel_prior(**kwargs)
    window_priors = gp.get_window_priors(**kwargs)
    priors.update(mean_priors)
    priors.update(kernel_priors)
    priors.update(window_priors)
    priors._resolve_conditions()
    return priors

