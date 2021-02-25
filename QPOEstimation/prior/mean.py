import bilby
import numpy as np
from bilby.core.prior import ConditionalPriorDict, Uniform, Beta

import QPOEstimation


def get_mean_prior(model_type, **kwargs):
    if model_type == 'polynomial':
        return get_polynomial_prior(**kwargs)
    elif model_type in _N_COMPONENT_PRIORS:
        return _N_COMPONENT_PRIORS[model_type](**kwargs)
    else:
        return dict()


def get_polynomial_prior(order=4, **kwargs):
    priors = bilby.core.prior.PriorDict()
    for i in range(order + 1):
        if kwargs['polynomial_max'] == 0:
            priors[f'mean:a{i}'] = 0
        else:
            priors[f'mean:a{i}'] = bilby.core.prior.Uniform(
                minimum=-kwargs['polynomial_max'], maximum=kwargs['polynomial_max'], name=f'mean:a{i}')
    return priors


def get_exponential_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = ConditionalPriorDict()
    for ii in range(n_components):
        if n_components == 1:
            priors[f'mean:tau_{ii}'] = Uniform(kwargs['tau_min'], kwargs['tau_max'], name=f"mean:tau_{ii}")
        elif ii == 0:
            priors[f"mean:tau_{ii}"] = Beta(minimum=kwargs['tau_min'], maximum=kwargs['tau_max'],
                                            alpha=1, beta=n_components, name=f"mean:tau_{ii}")
        else:
            priors[f"mean:tau_{ii}"] = QPOEstimation.prior.minimum.MinimumPrior(
                order=n_components - ii, minimum_spacing=minimum_spacing, minimum=kwargs['tau_min'],
                maximum=kwargs['tau_max'], name=f"mean:tau_{ii}")
        priors[f'mean:amplitude_{ii}'] = bilby.core.prior.LogUniform(minimum=kwargs['amplitude_min'],
                                                                     maximum=kwargs['amplitude_max'], name=f'A_{ii}')
        priors[f'mean:offset_{ii}'] = bilby.core.prior.LogUniform(minimum=kwargs['offset_min'],
                                                                  maximum=kwargs['offset_max'], name=f'sigma_{ii}')
    return priors


def get_gaussian_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = ConditionalPriorDict()
    for ii in range(n_components):
        if n_components == 1:
            priors[f'mean:t_0_{ii}'] = Uniform(kwargs['t_0_min'], kwargs['t_0_max'], name=f"mean:t_0_{ii}")
        elif ii == 0:
            priors[f"mean:t_0_{ii}"] = Beta(minimum=kwargs['t_0_min'], maximum=kwargs['t_0_max'], alpha=1,
                                            beta=n_components, name=f"mean:t_0_{ii}")
        else:
            priors[f"mean:t_0_{ii}"] = QPOEstimation.prior.minimum.MinimumPrior(
                order=n_components - ii, minimum_spacing=minimum_spacing, minimum=kwargs['t_0_min'],
                maximum=kwargs['t_0_max'], name=f"mean:t_0_{ii}")
        priors[f'mean:amplitude_{ii}'] = bilby.core.prior.LogUniform(
            minimum=kwargs['amplitude_min'], maximum=kwargs['amplitude_max'], name=f'A_{ii}')
        priors[f'mean:sigma_{ii}'] = bilby.core.prior.LogUniform(
            minimum=kwargs['sigma_min'], maximum=kwargs['sigma_max'], name=f'sigma_{ii}')
    return priors


def get_log_normal_priors(n_components=1, minimum_spacing=0, **kwargs):
    return get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)


def get_lorentzian_prior(n_components=1, minimum_spacing=0, **kwargs):
    return get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)


def get_fred_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)
    for ii in range(n_components):
        priors[f'mean:skewness_{ii}'] = bilby.core.prior.LogUniform(
            minimum=kwargs['skewness_min'], maximum=kwargs['skewness_max'], name=f'skewness_{ii}')
    return priors


_N_COMPONENT_PRIORS = dict(exponential=get_exponential_priors, gaussian=get_gaussian_priors,
                           log_normal=get_log_normal_priors, lorentzian=get_lorentzian_prior,
                           fred=get_fred_priors)