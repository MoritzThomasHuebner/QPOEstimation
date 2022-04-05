import numpy as np

import bilby

from . import minimum, psd, gp, mean


def get_priors(**kwargs) -> dict:
    """ Catch-all prior getter.

    Parameters
    ----------
    kwargs:
        All possible keyword arguments to construct the prior.

    Returns
    -------
    The Prior dict.

    """

    segment_length = kwargs["times"][-1] - kwargs["times"][0]
    sampling_frequency = 1 / (kwargs["times"][1] - kwargs["times"][0])

    if kwargs["band_minimum"] is None:
        kwargs["band_minimum"] = 2 / segment_length

    if kwargs["band_maximum"] is None:
        kwargs["band_maximum"] = sampling_frequency / 2

    if kwargs["min_log_c_red_noise"] is None:
        kwargs["min_log_c_red_noise"] = np.log(1 / segment_length)
    if kwargs["min_log_c_qpo"] is None:
        kwargs["min_log_c_qpo"] = np.log(1 / 10 / segment_length)
    if kwargs["max_log_c_red_noise"] is None:
        kwargs["max_log_c_red_noise"] = np.log(sampling_frequency)
    if kwargs["max_log_c_qpo"] is None:
        kwargs["max_log_c_qpo"] = np.log(kwargs.get("band_maximum", 1/2/segment_length))

    minimum = np.min(kwargs["y"]) if kwargs.get("offset", False) else 0
    maximum = np.max(kwargs["y"])
    span = maximum - minimum

    if kwargs["min_log_a"] is None:
        if kwargs["yerr"] is not None:
            kwargs["min_log_a"] = np.log(min(kwargs["yerr"]))
        else:
            kwargs["min_log_a"] = np.log(0.1 * span)
        if np.isinf(kwargs["min_log_a"]):
            kwargs["min_log_a"] = np.log(0.1 * span)
    if kwargs["max_log_a"] is None:
        kwargs["max_log_a"] = np.log(2*span)

    if kwargs["t_0_min"] is None:
        kwargs["t_0_min"] = kwargs["times"][0] - 0.1 * segment_length

    if kwargs["t_0_max"] is None:
        kwargs["t_0_max"] = kwargs["times"][-1] + 0.1 * segment_length

    if kwargs["sigma_min"] is None:
        kwargs["sigma_min"] = 0.5 * 1 / sampling_frequency

    if kwargs["sigma_max"] is None:
        kwargs["sigma_max"] = 2 * segment_length

    priors = bilby.core.prior.ConditionalPriorDict()
    mean_priors = mean.get_mean_prior(**kwargs)
    kernel_priors = gp.get_kernel_prior(**kwargs)
    window_priors = gp._get_window_priors(**kwargs)
    priors.update(mean_priors)
    priors.update(kernel_priors)
    priors.update(window_priors)
    priors._resolve_conditions()
    priors.conversion_function = gp.decay_constraint_conversion_function
    return priors
