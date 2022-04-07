import numpy as np

"""
This module contains a number of smoothing functions. 
Using these in the search for QPOs is probably not a good idea.
See Auchere et al. 2016.
"""


def moving_average(ys: np.ndarray, n: int = 64) -> np.ndarray:
    """ A simple moving-average filter.

    Parameters
    ----------
    ys: The data points are assumed to be equally spaced.
    n: The number of data points left and right of the current data point to use for the average.

    Returns
    -------
    The smoothed array.
    """
    res = np.zeros(len(ys))
    for i in range(len(ys)):
        if i < n:
            total_bins = i + n
            res[i] = np.mean(ys[:total_bins])
        elif i > len(ys) - n:
            total_bins = i - n
            res[i] = np.mean(ys[total_bins:])
        else:
            indices = np.arange(i - n, i + n, 1)
            res[i] = np.mean(ys[indices])
    return res


def exponential_smoothing(ys, alpha):
    """ A simple exponential filter.

    Parameters
    ----------
    ys: The data points are assumed to be equally spaced.
    alpha: The smoothing parameter in the interval [0, 1]. alpha = 1 is returns the initial time series exactly.

    Returns
    -------
    The smoothed array.
    """
    s = np.zeros(len(ys))
    s[0] = ys[0]
    for i in range(1, len(ys)):
        s[i] = alpha * ys[i] + s[i-1]*(1-alpha)
    return s


def second_order_exponential_smoothing(ys, alpha, beta):
    """ A second-order exponential filter.

    Parameters
    ----------
    ys: The data points are assumed to be equally spaced.
    alpha: The first smoothing parameter in the interval [0, 1].
    beta: The second smoothing parameter in the interval [0, 1].

    Returns
    -------
    The smoothed array.
    """
    s = np.zeros(len(ys))
    b = np.zeros(len(ys))
    s[0] = ys[0]
    b[0] = ys[1] - ys[0]
    for i in range(1, len(ys)):
        s[i] = alpha * ys[i] + (1 - alpha)*(s[i-1] - b[i-1])
        b[i] = beta * (s[i] - s[i-1]) + (1 - beta)*b[i-1]
    return s


def two_sided_exponential_smoothing(ys, alpha):
    """ Applies the simple exponential filter in both directions and takes the average.
    Exponential filtering is not symmetric and the smoothed time series often trails the actual data somewhat.

    Parameters
    ----------
    ys: The data points are assumed to be equally spaced.
    alpha: The smoothing parameter in the interval [0, 1]. alpha = 1 is returns the initial time series exactly.

    Returns
    -------
    The smoothed array.
    """

    forward = exponential_smoothing(ys, alpha)
    backwards = exponential_smoothing(ys[::-1], alpha)[::-1]
    return (forward + backwards)/2


def two_sided_second_order_exponential_smoothing(ys, alpha, beta):
    """ Applies the second-order exponential filter in both directions and takes the average.
    Exponential filtering is not symmetric and the smoothed time series often trails the actual data somewhat.

    Parameters
    ----------
    ys: The data points are assumed to be equally spaced.
    alpha: The smoothing parameter in the interval [0, 1].
    beta: The second smoothing parameter in the interval [0, 1].

    Returns
    -------
    The smoothed array.
    """
    forward = second_order_exponential_smoothing(ys, alpha, beta)
    backwards = second_order_exponential_smoothing(ys[::-1], alpha, beta)[::-1]
    return (forward + backwards)/2


def boxcar_filter(ys, n):
    """ A simple boxcar filter.

    Parameters
    ----------
    ys: The data points are assumed to be equally spaced.
    n: The number of data points left and right of the current data point to use for the average.

    Returns
    -------
    The smoothed array.
    """

    res = np.zeros(len(ys))
    for i, c in enumerate(ys):
        if i < (n - 1)/2:
            boxcar = ys[0: int(i + (n - 1) / 2)]
        elif i > len(ys) - (n - 1)/2:
            boxcar = ys[int(i - (n - 1) / 2):-1]
        else:
            boxcar = ys[int(i - (n - 1) / 2): int(i + (n - 1) / 2)]
        res[i] = sum(boxcar) / len(boxcar)
    return res
