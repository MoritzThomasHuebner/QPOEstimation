import bilby
import numpy as np
from bilby.core.prior import ConditionalPriorDict, Uniform, Beta, DeltaFunction

import QPOEstimation


def get_mean_prior(model_type, **kwargs):
    minimum = np.min(kwargs['y']) if kwargs.get('offset', False) else 0
    maximum = np.max(kwargs['y'])
    span = maximum - minimum
    if kwargs['amplitude_min'] is None:
        kwargs['amplitude_min'] = 0.1 * span
    if kwargs['amplitude_max'] is None:
        kwargs['amplitude_max'] = 2 * span

    if model_type == 'polynomial':
        priors = get_polynomial_prior(**kwargs)
    elif model_type in _N_COMPONENT_PRIORS:
        priors = _N_COMPONENT_PRIORS[model_type](**kwargs)
    elif model_type in _PIECEWISE_PRIORS:
        return _PIECEWISE_PRIORS[model_type](**kwargs)
    else:
        priors = dict()

    offset_min = kwargs.get('offset_min')
    offset_max = kwargs.get('offset_max')
    if offset_min is None:
        offset_min = minimum
    if offset_max is None:
        offset_max = maximum

    if kwargs.get('offset', False):
        priors['mean:offset'] = bilby.prior.Uniform(minimum=offset_min, maximum=offset_max,
                                                    name="log offset")
    return priors


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
        priors[f'mean:log_amplitude_{ii}'] = bilby.core.prior.Uniform(
            minimum=np.log(kwargs['amplitude_min']),
            maximum=np.log(kwargs['amplitude_max']), name=f'ln A_{ii}')
    return priors


def get_gaussian_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = ConditionalPriorDict()
    for ii in range(n_components):
        if kwargs['t_0_min'] == kwargs['t_0_max']:
            priors[f'mean:t_0_{ii}'] = DeltaFunction(kwargs['t_0_min'], name=f"mean:t_0_{ii}")
        elif n_components == 1:
            priors[f'mean:t_0_{ii}'] = Uniform(kwargs['t_0_min'], kwargs['t_0_max'], name=f"mean:t_0_{ii}")
        elif ii == 0:
            priors[f"mean:t_0_{ii}"] = Beta(minimum=kwargs['t_0_min'], maximum=kwargs['t_0_max'], alpha=1,
                                            beta=n_components, name=f"mean:t_0_{ii}")
        else:
            priors[f"mean:t_0_{ii}"] = QPOEstimation.prior.minimum.MinimumPrior(
                order=n_components - ii, minimum_spacing=minimum_spacing, minimum=kwargs['t_0_min'],
                maximum=kwargs['t_0_max'], name=f"mean:t_0_{ii}")
        priors[f'mean:log_amplitude_{ii}'] = bilby.core.prior.Uniform(
            minimum=np.log(kwargs['amplitude_min']), maximum=np.log(kwargs['amplitude_max']), name=f'ln A_{ii}')
        priors[f'mean:log_sigma_{ii}'] = bilby.core.prior.Uniform(
            minimum=np.log(kwargs['sigma_min']), maximum=np.log(kwargs['sigma_max']), name=f'ln sigma_{ii}')
    return priors


def get_skew_gaussian_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)
    for ii in range(n_components):
        del priors[f'mean:log_sigma_{ii}']
        priors[f'mean:log_sigma_rise_{ii}'] = bilby.core.prior.Uniform(
            minimum=np.log(kwargs['sigma_min']), maximum=np.log(kwargs['sigma_max']), name=f'ln sigma_rise_{ii}')
        priors[f'mean:log_sigma_fall_{ii}'] = bilby.core.prior.Uniform(
            minimum=np.log(kwargs['sigma_min']), maximum=np.log(kwargs['sigma_max']), name=f'ln sigma_fall_{ii}')
    return priors


def get_log_normal_priors(n_components=1, minimum_spacing=0, **kwargs):
    return get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)


def get_lorentzian_prior(n_components=1, minimum_spacing=0, **kwargs):
    return get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)


def get_fred_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)
    for p in list(priors.keys()):
        if 'sigma' in p:
            del priors[p]
    for ii in range(n_components):
        duration = kwargs['times'][-1] - kwargs['times'][0]
        dt = kwargs['times'][1] - kwargs['times'][0]
        sigma_min = kwargs.get("sigma_min")
        sigma_max = kwargs.get("sigma_max")
        if sigma_min is None:
            sigma_min = dt
        if sigma_max is None:
            sigma_max = duration
        priors[f'mean:log_sigma_rise_{ii}'] = bilby.core.prior.Uniform(
            minimum=np.log(sigma_min), maximum=np.log(sigma_max), name=f'ln sigma_rise_{ii}')
        priors[f'mean:log_sigma_fall_{ii}'] = bilby.core.prior.Uniform(
            minimum=np.log(sigma_min), maximum=np.log(sigma_max), name=f'ln sigma_fall_{ii}')
    return priors


def get_fred_norris_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)
    for ii in range(n_components):
        del priors[f'mean:log_sigma_{ii}']
        priors[f'mean:log_psi_{ii}'] = bilby.core.prior.Uniform(minimum=np.log(2e-2),
                                                                maximum=np.log(2e2), name=f'psi_{ii}')
        priors[f'mean:delta_{ii}'] = bilby.core.prior.Uniform(minimum=0, maximum=kwargs['times'][-1],
                                                              name=f'delta_{ii}')
    return priors


def get_fred_norris_extended_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = get_fred_norris_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)
    for ii in range(n_components):
        priors[f'mean:log_gamma_{ii}'] = bilby.core.prior.Uniform(minimum=np.log(1e-3), maximum=np.log(1e3),
                                                                  name=f'log_gamma_{ii}')
        priors[f'mean:log_nu_{ii}'] = bilby.core.prior.Uniform(minimum=np.log(1e-3), maximum=np.log(1e3),
                                                               name=f'log_nu_{ii}')
    return priors


def get_piecewise_linear_priors(n_components, minimum_spacing, **kwargs):
    priors = bilby.core.prior.ConditionalPriorDict()
    for i in range(n_components):
        priors[f"mean:beta_{i}"] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name=f"beta_{i}")

    t_0_min = kwargs["times"][0]
    t_0_max = kwargs["times"][-1]

    for i in range(1, n_components):
        loc = (t_0_max - t_0_min) * i / n_components + t_0_min
        priors[f"mean:k_{i}"] = DeltaFunction(peak=loc, name=f"mean:k_{i}")
    # for i in range(2, n_components):
    #     if i == 2:
    #         priors[f"mean:k_{i}"] = Beta(minimum=kwargs['t_0_min'], maximum=kwargs['t_0_max'],
    #                                      alpha=1, beta=n_components-2, name=f"mean:k_{i}")
    #     else:
    #         priors[f"mean:k_{i}"] = QPOEstimation.prior.minimum.MinimumPrior(
    #             order=n_components - i, minimum_spacing=minimum_spacing, minimum=kwargs['t_0_min'],
    #             maximum=kwargs['t_0_max'], name=f"mean:k_{i}")
    priors._resolve_conditions()
    return priors


def get_piecewise_cubic_priors(n_components, minimum_spacing, **kwargs):
    priors = bilby.core.prior.ConditionalPriorDict()
    maximum = 1
    priors[f"mean:alpha_0"] = bilby.core.prior.Uniform(minimum=-maximum, maximum=maximum, name=f"alpha_0")
    priors[f"mean:beta_0"] = bilby.core.prior.Uniform(minimum=-maximum, maximum=maximum, name=f"beta_0")
    priors[f"mean:gamma_0"] = bilby.core.prior.Uniform(minimum=-maximum, maximum=maximum, name=f"gamma_0")
    for i in range(n_components):
        priors[f"mean:delta_{i}"] = bilby.core.prior.Uniform(minimum=-maximum, maximum=maximum, name=f"delta_{i}")

    t_0_min = kwargs["times"][0]
    t_0_max = kwargs["times"][-1]
    for i in range(1, n_components):
        loc = (t_0_max - t_0_min) * i / n_components + t_0_min
        priors[f"mean:k_{i}"] = DeltaFunction(peak=loc, name=f"mean:k_{i}")

        # if n_components == 2:
        #     priors[f"mean:k_{i}"] = Uniform(minimum=t_0_min, maximum=t_0_max, name=f"mean:k_{i}")
        #     break
        # elif i == 1:
        #     priors[f"mean:k_{i}"] = Beta(minimum=t_0_min, maximum=t_0_max,
        #                                  alpha=1, beta=n_components, name=f"mean:k_{i}")
        # else:
        #     priors[f"mean:k_{i}"] = QPOEstimation.prior.minimum.MinimumPrior(
        #         order=n_components, minimum_spacing=minimum_spacing, minimum=t_0_min,
        #         maximum=t_0_max, name=f"mean:k_{i}")
    priors._resolve_conditions()
    return priors

_N_COMPONENT_PRIORS = dict(exponential=get_exponential_priors, gaussian=get_gaussian_priors,
                           log_normal=get_log_normal_priors, lorentzian=get_lorentzian_prior,
                           fred=get_fred_priors, skew_gaussian=get_skew_gaussian_priors,
                           fred_norris=get_fred_norris_priors, fred_norris_extended=get_fred_norris_extended_priors)

_PIECEWISE_PRIORS = dict(piecewise_linear=get_piecewise_linear_priors, piecewise_cubic=get_piecewise_cubic_priors)
