import bilby
import numpy as np
from bilby.core.prior import ConditionalPriorDict, Uniform, Beta

import QPOEstimation


def get_polynomial_prior(polynomial_max=10, order=4):
    priors = bilby.core.prior.PriorDict()
    for i in range(order + 1):
        if polynomial_max == 0:
            priors[f'mean:a{i}'] = 0
        else:
            priors[f'mean:a{i}'] = bilby.core.prior.Uniform(
                minimum=-polynomial_max, maximum=polynomial_max, name=f'mean:a{i}')
    return priors


def get_mean_prior(model_type, **kwargs):
    if model_type == 'polynomial':
        return get_polynomial_prior(polynomial_max=kwargs['polynomial_max'])
    else:
        return dict()


def get_fred_priors(n_freds=1, t_min=0, t_max=2000, minimum_spacing=0):
    priors = ConditionalPriorDict()

    if n_freds == 1:
        priors[f'mean:amplitude_0'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e12, name='A')
        priors[f'mean:sigma_0'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=10000, name='sigma')
        priors[f'mean:skewness_0'] = bilby.core.prior.LogUniform(minimum=np.exp(-20), maximum=np.exp(20),
                                                                    name=f'skewness_0')
        priors[f'mean:t_max_0'] = Uniform(t_min, t_max, name="t_max")
        return priors
    for ii in range(n_freds):
        if ii == 0:
            priors[f"mean:t_max_{ii}"] = Beta(minimum=t_min, maximum=t_max, alpha=1, beta=n_freds, name=f"mean:t_max_{ii}")
        else:
            priors[f"mean:t_max_{ii}"] = QPOEstimation.prior.minimum.MinimumPrior(
                order=n_freds-ii, minimum_spacing=minimum_spacing, minimum=t_min, maximum=t_max, name=f"mean:t_max_{ii}")
        priors[f'mean:amplitude_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e12, name=f'A_{ii}')
        priors[f'mean:sigma_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=10000, name=f'sigma_{ii}')
        priors[f'mean:skewness_{ii}'] = bilby.core.prior.LogUniform(minimum=np.exp(-20), maximum=np.exp(20), name=f'log skewness_{ii}')
        priors[f"mean:t_max_{ii}"].__class__.__name__ = "MinimumPrior"
    return priors