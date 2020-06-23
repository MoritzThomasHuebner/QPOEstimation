import numpy as np
from scipy.stats import poisson


def poisson_process(times, func, **func_params):
    dt = times[1] - times[0]
    rates = func(times, **func_params) * dt
    counts = poisson.rvs(rates)
    return counts


def tte_poisson_process(t_start, t_end, func, resolution_limit=125e-6, **func_params):
    times = np.arange(t_start, t_end, resolution_limit)
    counts = poisson_process(times=times, func=func, **func_params)
    print(len(np.where(counts >= 1))/len(counts))
    ttes = np.asarray(np.where(counts >= 1)) * resolution_limit
    return ttes[0]
