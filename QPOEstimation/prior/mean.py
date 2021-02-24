import bilby
import numpy as np
from bilby.core.prior import ConditionalPriorDict, Uniform, Beta

import QPOEstimation


def get_mean_prior(model_type, **kwargs):
    if model_type == 'polynomial':
        return get_polynomial_prior(polynomial_max=kwargs['polynomial_max'])
    elif model_type in _N_COMPONENT_PRIORS:
        return _N_COMPONENT_PRIORS[model_type](**kwargs)
    else:
        return dict()


def get_polynomial_prior(polynomial_max=10, order=4, **kwargs):
    priors = bilby.core.prior.PriorDict()
    for i in range(order + 1):
        if polynomial_max == 0:
            priors[f'mean:a{i}'] = 0
        else:
            priors[f'mean:a{i}'] = bilby.core.prior.Uniform(
                minimum=-polynomial_max, maximum=polynomial_max, name=f'mean:a{i}')
    return priors


def get_exponential_priors(n_components=1, t_min=-2000, t_max=2000, minimum_spacing=0, **kwargs):
    priors = ConditionalPriorDict()
    for ii in range(n_components):
        if n_components == 1:
            priors[f'mean:tau_{ii}'] = Uniform(t_min, t_max, name=f"mean:tau_{ii}")
        elif ii == 0:
            priors[f"mean:tau_{ii}"] = Beta(minimum=t_min, maximum=t_max, alpha=1, beta=n_components,
                                            name=f"mean:tau_{ii}")
        else:
            priors[f"mean:tau_{ii}"] = QPOEstimation.prior.minimum.MinimumPrior(
                order=n_components - ii, minimum_spacing=minimum_spacing, minimum=t_min, maximum=t_max,
                name=f"mean:tau_{ii}")
        priors[f'mean:amplitude_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-12, maximum=1e12, name=f'A_{ii}')
        priors[f'mean:offset_{ii}'] = bilby.core.prior.LogUniform(minimum=-1e12, maximum=1e12, name=f'sigma_{ii}')
        # priors[f"mean:tau_{ii}"].__class__.__name__ = "MinimumPrior"
    return priors


def get_gaussian_priors(n_components=1, t_min=0, t_max=2000, minimum_spacing=0, **kwargs):
    priors = ConditionalPriorDict()
    for ii in range(n_components):
        if n_components == 1:
            priors[f'mean:t_0_{ii}'] = Uniform(t_min, t_max, name=f"mean:t_0_{ii}")
        elif ii == 0:
            priors[f"mean:t_0_{ii}"] = Beta(minimum=t_min, maximum=t_max, alpha=1, beta=n_components, name=f"mean:t_0_{ii}")
        else:
            priors[f"mean:t_0_{ii}"] = QPOEstimation.prior.minimum.MinimumPrior(
                order=n_components - ii, minimum_spacing=minimum_spacing, minimum=t_min, maximum=t_max, name=f"mean:t_0_{ii}")
        priors[f'mean:amplitude_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-12, maximum=1e12, name=f'A_{ii}')
        priors[f'mean:sigma_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-12, maximum=1e12, name=f'sigma_{ii}')
        # priors[f"mean:t_max_{ii}"].__class__.__name__ = "MinimumPrior"
    return priors


def get_log_normal_priors(n_components=1, t_min=0, t_max=2000, minimum_spacing=0, **kwargs):
    return get_gaussian_priors(n_components=n_components, t_min=t_min, t_max=t_max, minimum_spacing=minimum_spacing)


def get_lorentzian_prior(n_components=1, t_min=0, t_max=2000, minimum_spacing=0, **kwargs):
    return get_gaussian_priors(n_components=n_components, t_min=t_min, t_max=t_max, minimum_spacing=minimum_spacing)


def get_fred_priors(n_components=1, t_min=0, t_max=2000, minimum_spacing=0, **kwargs):
    priors = get_gaussian_priors(n_components=n_components, t_min=t_min, t_max=t_max, minimum_spacing=minimum_spacing)
    for ii in range(n_components):
        priors[f'mean:skewness_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-12, maximum=1e12, name=f'log skewness_{ii}')
        # priors[f"mean:t_max_{ii}"].__class__.__name__ = "MinimumPrior"
    return priors


_N_COMPONENT_PRIORS = dict(exponential=get_exponential_priors, gaussian=get_gaussian_priors,
                           log_normal=get_log_normal_priors, lorentzian=get_lorentzian_prior, fred=get_fred_priors)


