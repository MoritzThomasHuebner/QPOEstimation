from copy import deepcopy
import numpy as np


def anscombe(k):
    rate = deepcopy(k)
    # rate[np.where(rate == 0)] = 1
    return 2*np.sqrt(rate + 0.375)


def bar_lev(k):
    rate = deepcopy(k)
    # rate[np.where(rate == 0)] = 1
    return np.sqrt(rate + 1) + np.sqrt(rate)
