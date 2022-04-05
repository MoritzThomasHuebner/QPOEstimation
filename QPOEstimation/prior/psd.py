import bilby
import numpy as np


def get_red_noise_prior(**kwargs):
    """ Provides a prior dict for the red noise PSD model.

    Parameters
    ----------
    kwargs:
        Catch all for all relevant prior bounds.

    Returns
    -------
    The prior dict.

    """
    sigma_min = kwargs.get("sigma_min", np.exp(-30))
    sigma_max = kwargs.get("sigma_max", np.exp(30))
    if sigma_min is None:
        sigma_min = np.exp(-30)
    if sigma_max is None:
        sigma_max = np.exp(30)

    prior = bilby.core.prior.ConditionalPriorDict()
    prior["alpha"] = bilby.core.prior.Uniform(0, 20, name="alpha")
    prior["log_beta"] = bilby.core.prior.Uniform(-100, 100, name="log_beta")
    if sigma_min == sigma_max:
        prior["log_sigma"] = bilby.core.prior.DeltaFunction(np.log(sigma_min), name="log_sigma")
    else:
        prior["log_sigma"] = bilby.core.prior.Uniform(np.log(sigma_min), np.log(sigma_max), name="log_sigma")
    return prior


def log_width_condition_func(reference_params: list, log_frequency: float) -> dict:
    """ Ensures the QPO is not wider than half the QPO frequency.

    Parameters
    ----------
    reference_params:
        Unused parameter. Necessary to fit the bilby interface.
    log_frequency:
        The QPO log frequency.

    Returns
    -------
    The dictionary with maximum of the log width prior.
    """
    frequency = np.exp(log_frequency)
    width_max = frequency/2
    log_width_max = np.log(width_max)
    return dict(maximum=log_width_max)


def get_qpo_prior(frequencies: np.ndarray = None, **kwargs) -> dict:
    """ Provides a QPO prior dict.

    Parameters
    ----------
    frequencies:
        The periodogram frequencies.
    kwargs:
        Additional parameters.

    Returns
    -------
    The prior dict.
    """

    if frequencies is None:
        df = 1
        max_frequency = 512
    else:
        df = frequencies[1] - frequencies[0]
        max_frequency = frequencies[-1]
    prior = bilby.core.prior.ConditionalPriorDict()
    prior["log_amplitude"] = bilby.core.prior.Uniform(-30, 30, name="log_amplitude")
    prior["log_width"] = bilby.core.prior.ConditionalUniform(condition_func=log_width_condition_func,
        minimum=np.log(df/np.pi), maximum=kwargs.get("max_log_width", np.log(0.25*max_frequency)), name="log_width")
    prior["log_frequency"] = bilby.core.prior.Uniform(
        kwargs.get("min_log_f", np.log(2*df)),  kwargs.get("max_log_f", np.log(max_frequency)), name="log_frequency")
    prior._resolve_conditions()
    return prior


def broken_power_law_conversion_function(params, **kwargs):
    new_params = params.copy()
    new_params["alpha_diffs"] = new_params["alpha_1"] - new_params["alpha_2"]
    return new_params


def get_broken_power_law_prior(frequencies=None):
    """ Provides a broken power-law prior dict.

    Parameters
    ----------
    frequencies:
        The periodogram frequencies.

    Returns
    -------
    The prior dict.
    """

    prior = bilby.core.prior.PriorDict()
    prior["alpha_1"] = bilby.core.prior.Uniform(0, 10, name="alpha_1")
    prior["alpha_2"] = bilby.core.prior.Uniform(0, 10, name="alpha_2")
    if frequencies is None:
        prior["log_delta"] = bilby.core.prior.Uniform(-30, 30, name="log_delta")
    else:
        prior["log_delta"] = bilby.core.prior.Uniform(np.log(frequencies[1]), np.log(frequencies[-1]), name="log_delta")
    prior["rho"] = bilby.core.prior.DeltaFunction(peak=-1)
    prior["log_beta"] = bilby.core.prior.Uniform(-60, 60, name="log_beta")
    prior["log_sigma"] = bilby.core.prior.Uniform(-30, 30, name="log_sigma")
    prior["alpha_diffs"] = bilby.core.prior.Constraint(0, 1000, name="alpha_diffs")
    prior.conversion_function = broken_power_law_conversion_function
    return prior


def get_full_prior(noise_model="red_noise", frequencies=None):
    """ Provides a combined (broken) power-law and QPO prior dict.

    Parameters
    ----------
    noise_model:
        Needs to be from ['red_noise', 'broken_power_law']
    frequencies:
        The periodogram frequencies.

    Returns
    -------
    The prior dict.
    """
    if noise_model == "broken_power_law":
        noise_prior = get_broken_power_law_prior()
    else:
        noise_prior = get_red_noise_prior()
    prior = get_qpo_prior(frequencies=frequencies)
    for key, value in noise_prior.items():
        prior[key] = value
    return prior
