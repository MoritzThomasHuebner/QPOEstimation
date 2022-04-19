import math

import bilby
import numpy as np
from bilby.core.prior import ConditionalPriorDict, Uniform, Beta, DeltaFunction

import QPOEstimation


def get_mean_prior(model_type: str, **kwargs) -> dict:
    """

    Parameters
    ----------
    model_type:
        Must be a kew from `_N_COMPONENT_PRIORS`
    kwargs:
        Any other keywords to set prior boundaries.

    Returns
    -------
    The Prior dict.

    """
    minimum = np.min(kwargs["y"]) if kwargs.get("offset", False) else 0
    maximum = np.max(kwargs["y"])
    span = maximum - minimum
    if kwargs["amplitude_min"] is None:
        kwargs["amplitude_min"] = 0.1 * span
    if kwargs["amplitude_max"] is None:
        kwargs["amplitude_max"] = 2 * span

    if model_type == "polynomial":
        priors = _get_polynomial_prior(**kwargs)
    elif model_type in _N_COMPONENT_PRIORS:
        priors = _N_COMPONENT_PRIORS[model_type](**kwargs)
    else:
        priors = dict()

    offset_min = kwargs.get("offset_min")
    offset_max = kwargs.get("offset_max")
    if offset_min is None:
        offset_min = minimum
    if offset_max is None:
        offset_max = maximum

    if kwargs.get("offset", False):
        if math.isclose(offset_min, offset_max):
            priors["mean:offset"] = bilby.prior.DeltaFunction(peak=offset_max, name="log offset")
        else:
            priors["mean:offset"] = bilby.prior.Uniform(minimum=offset_min, maximum=offset_max,
                                                        name="offset")
    return priors


def _get_polynomial_prior(n_components=4, **kwargs):
    priors = bilby.core.prior.PriorDict()
    for i in range(n_components):
        if kwargs["polynomial_max"] == 0:
            priors[f"mean:a{i}"] = 0
        else:
            priors[f"mean:a{i}"] = bilby.core.prior.Uniform(
                minimum=-kwargs["polynomial_max"], maximum=kwargs["polynomial_max"], name=f"mean:a{i}")
    return priors


def _get_gaussian_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = ConditionalPriorDict()
    for ii in range(n_components):
        if math.isclose(kwargs["t_0_min"], kwargs["t_0_max"]):
            priors[f"mean:t_0_{ii}"] = DeltaFunction(kwargs["t_0_min"], name=f"mean:t_0_{ii}")
        elif n_components == 1:
            priors[f"mean:t_0_{ii}"] = Uniform(kwargs["t_0_min"], kwargs["t_0_max"], name=f"mean:t_0_{ii}")
        elif ii == 0:
            priors[f"mean:t_0_{ii}"] = Beta(minimum=kwargs["t_0_min"], maximum=kwargs["t_0_max"], alpha=1,
                                            beta=n_components, name=f"mean:t_0_{ii}")
        else:
            priors[f"mean:t_0_{ii}"] = QPOEstimation.prior.minimum.MinimumPrior(
                order=n_components - ii, minimum_spacing=minimum_spacing, minimum=kwargs["t_0_min"],
                maximum=kwargs["t_0_max"], name=f"mean:t_0_{ii}")

        if math.isclose(np.log(kwargs["amplitude_min"]), np.log(kwargs["amplitude_max"])):
            priors[f"mean:log_amplitude_{ii}"] = \
                bilby.prior.DeltaFunction(peak=np.log(kwargs["amplitude_max"]), name=f"ln A_{ii}")
        else:
            priors[f"mean:log_amplitude_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(kwargs["amplitude_min"]),
                maximum=np.log(kwargs["amplitude_max"]), name=f"ln A_{ii}")

        if math.isclose(np.log(kwargs["sigma_min"]), np.log(kwargs["sigma_max"])):
            priors[f"mean:log_sigma_{ii}"] = \
                bilby.prior.DeltaFunction(peak=np.log(kwargs["sigma_max"]), name=f"ln sigma_{ii}")
        else:
            priors[f"mean:log_sigma_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(kwargs["sigma_min"]),
                maximum=np.log(kwargs["sigma_max"]), name=f"ln sigma_{ii}")
    return priors


def _get_skew_gaussian_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = _get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)
    for ii in range(n_components):
        del priors[f"mean:log_sigma_{ii}"]
        if math.isclose(np.log(kwargs["sigma_min"]), np.log(kwargs["sigma_max"])):
            priors[f"mean:log_sigma_rise_{ii}"] = bilby.core.prior.DeltaFunction(
                peak=np.log(kwargs["sigma_max"]), name=f"ln sigma_rise_{ii}")
            priors[f"mean:log_sigma_fall_{ii}"] = bilby.core.prior.DeltaFunction(
                peak=np.log(kwargs["sigma_max"]), name=f"ln sigma_fall_{ii}")
        else:
            priors[f"mean:log_sigma_rise_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(kwargs["sigma_min"]), maximum=np.log(kwargs["sigma_max"]), name=f"ln sigma_rise_{ii}")
            priors[f"mean:log_sigma_fall_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(kwargs["sigma_min"]), maximum=np.log(kwargs["sigma_max"]), name=f"ln sigma_fall_{ii}")

    return priors


def _get_log_normal_priors(n_components=1, minimum_spacing=0, **kwargs):
    return _get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)


def _get_lorentzian_prior(n_components=1, minimum_spacing=0, **kwargs):
    return _get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)


def _get_skew_exponential_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = _get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)
    for p in list(priors.keys()):
        if "sigma" in p:
            del priors[p]
    for ii in range(n_components):
        duration = kwargs["times"][-1] - kwargs["times"][0]
        dt = kwargs["times"][1] - kwargs["times"][0]
        sigma_min = kwargs.get("sigma_min")
        sigma_max = kwargs.get("sigma_max")
        if sigma_min is None:
            sigma_min = dt
        if sigma_max is None:
            sigma_max = duration
        if math.isclose(sigma_min, sigma_max):
            priors[f"mean:log_sigma_rise_{ii}"] = bilby.core.prior.DeltaFunction(
                peak=np.log(sigma_max), name=f"ln sigma_rise_{ii}")
            priors[f"mean:log_sigma_fall_{ii}"] = bilby.core.prior.DeltaFunction(
                peak=np.log(sigma_max), name=f"ln sigma_fall_{ii}")
        else:
            priors[f"mean:log_sigma_rise_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(sigma_min), maximum=np.log(sigma_max), name=f"ln sigma_rise_{ii}")
            priors[f"mean:log_sigma_fall_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(sigma_min), maximum=np.log(sigma_max), name=f"ln sigma_fall_{ii}")
    return priors


def _get_fred_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = _get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)
    for ii in range(n_components):
        del priors[f"mean:log_sigma_{ii}"]
        priors[f"mean:log_psi_{ii}"] = bilby.core.prior.Uniform(minimum=np.log(2e-2),
                                                                maximum=np.log(2e4), name=f"psi_{ii}")
        priors[f"mean:delta_{ii}"] = bilby.core.prior.Uniform(minimum=0, maximum=kwargs["times"][-1],
                                                              name=f"delta_{ii}")
    return priors


def _get_fred_extended_priors(n_components=1, minimum_spacing=0, **kwargs):
    priors = _get_fred_priors(n_components=n_components, minimum_spacing=minimum_spacing, **kwargs)
    for ii in range(n_components):
        priors[f"mean:log_gamma_{ii}"] = bilby.core.prior.Uniform(minimum=np.log(1e-3), maximum=np.log(1e3),
                                                                  name=f"log_gamma_{ii}")
        priors[f"mean:log_nu_{ii}"] = bilby.core.prior.Uniform(minimum=np.log(1e-3), maximum=np.log(1e3),
                                                               name=f"log_nu_{ii}")
    return priors


_N_COMPONENT_PRIORS = dict(
    gaussian=_get_gaussian_priors, log_normal=_get_log_normal_priors, lorentzian=_get_lorentzian_prior,
    skew_exponential=_get_skew_exponential_priors, skew_gaussian=_get_skew_gaussian_priors,
    fred=_get_fred_priors, fred_extended=_get_fred_extended_priors)
