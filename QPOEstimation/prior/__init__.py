from . import minimum, psd, gp, mean

import bilby


def get_priors(**kwargs):
    """ Catch all function"""
    priors = bilby.core.prior.ConditionalPriorDict()
    mean_priors = mean.get_mean_prior(**kwargs)
    kernel_priors = gp.get_kernel_prior(**kwargs)
    window_priors = gp.get_window_priors(**kwargs)
    priors.update(mean_priors)
    priors.update(kernel_priors)
    priors.update(window_priors)
    priors._resolve_conditions()
    return priors

