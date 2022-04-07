import numpy as np


def anscombe(k: np.ndarray) -> np.ndarray:
    """ Performs a variance stabilising Anscombe transformation.

    Parameters
    ----------
    k: An array of Poissonian data.

    Returns
    -------
    An array of approximately Gaussian data with sigma = 1.
    """
    return 2*np.sqrt(k + 0.375)


def bar_lev(k: np.ndarray) -> np.ndarray:
    """ Performs a variance stabilising Bar-Lev transformation.

    Parameters
    ----------
    k: An array of Poissonian data.

    Returns
    -------
    An array of approximately Gaussian data with sigma = 1.
    """
    return np.sqrt(k + 1) + np.sqrt(k)
