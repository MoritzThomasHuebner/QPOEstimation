import numpy as np
from scipy.stats import poisson


def poisson_process_norm(times: np.ndarray, func: callable, **func_params) -> np.ndarray:
    """ Creates normalised Poisson-process counts given a rate function `func`.

    Parameters
    ----------
    times:
        The location of the equally-spaced time bins.
    func:
        The rate function with the time coordinates as its first argument.
    func_params:
        The parameters of the rate function.

    Returns
    -------
    The Poisson process data.
    """
    dt = times[1] - times[0]
    rates = func(times, **func_params) * dt
    return poisson.rvs(rates)


def poisson_process(times: np.ndarray, func: callable, **func_params) -> np.ndarray:
    """ Creates unnormalised Poisson-process counts given a rate function `func`.
    This is effectively assuming the rate function provides the rate per time step in `times`.

    Parameters
    ----------
    times:
        The location of the equally-spaced time bins.
    func:
        The rate function with the time coordinates as its first argument.
    func_params:
        The parameters of the rate function.

    Returns
    -------
    The Poisson process data.
    """
    rates = func(times, **func_params)
    return poisson.rvs(rates)


def tte_poisson_process(
        t_start: float, t_end: float, func: callable, resolution_limit: float = 125e-6, **func_params) -> np.ndarray:
    """ Creates time-tagged events (TTEs) from a given rate function `func`.

    Parameters
    ----------
    t_start:
        The earliest possible TTE arrival.
    t_end:
        The lastest possible TTE arrival.
    func:
        The rate function with the time coordinates as its first argument.
    resolution_limit:
        Resolution limit of the TTEs. No two TTEs can occur within a bin at that size.
        Sort of implements dead time, but not accurately. Just set it to a value much smaller than actual rate.
        (Default_value = 125e-6)
    func_params:
        The parameters of the rate function.

    Returns
    -------
    An array with TTE time stamps.

    """
    times = np.arange(t_start, t_end, resolution_limit)
    counts = poisson_process_norm(times=times, func=func, **func_params)
    return (np.asarray(np.where(counts >= 1)) * resolution_limit)[0]
