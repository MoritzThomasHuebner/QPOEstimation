from copy import deepcopy
import numpy as np


def anscombe(k):
    return 2*np.sqrt(k + 0.375)


def bar_lev(k):
    return np.sqrt(k + 1) + np.sqrt(k)
