from copy import deepcopy

import numpy as np

from QPOEstimation.stabilisation import bar_lev


def burst_envelope(times, amplitude, t_max, sigma, skewness):
    if amplitude == 0:
        return np.zeros(len(times))
    before_burst_indices = np.where(times <= t_max)
    after_burst_indices = np.where(times > t_max)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = amplitude * np.exp((times[before_burst_indices] - t_max) / sigma)
    envelope[after_burst_indices] = amplitude * np.exp(-(times[after_burst_indices] - t_max) / sigma / skewness)
    return envelope


def qpo_shot(times, amplitude, frequency, t_qpo, phase, decay_time):
    if amplitude == 0:
        return np.zeros(len(times))
    res = np.zeros(len(times))
    idxs = [np.where(times >= t_qpo)][0]
    res[idxs] = amplitude*(1 + np.cos(2 * np.pi * frequency * (times[idxs] - t_qpo) + phase)) * np.exp(
        -(times[idxs] - t_qpo) / decay_time)
    return res


def zero_mean_qpo_shot(times, start_time, amplitude, decay_time, frequency, phase, **kwargs):
    t = deepcopy(times)
    t -= times[0]
    start_time -= times[0]
    qpo = np.zeros(len(t))
    if decay_time > 0:
        indices = np.where(t > start_time)
        qpo[indices] = amplitude * np.exp(-(t[indices] - start_time) / decay_time) * \
                       np.cos(2 * np.pi * frequency * (t[indices] - start_time) + phase)
    if decay_time <= 0:
        indices = np.where(t < start_time)
        qpo[indices] = amplitude * np.exp(-(t[indices] - start_time) / decay_time) * \
                       np.cos(2 * np.pi * frequency * (t[indices] - start_time) + phase)

    return qpo


def two_sided_qpo_shot(times, peak_time, amplitude, decay_time, frequency, phase, **kwargs):
    t = deepcopy(times)
    t -= times[0]
    peak_time -= times[0]
    qpo = np.zeros(len(t))
    falling_indices = np.where(t > peak_time)
    qpo[falling_indices] = amplitude * np.exp(-(t[falling_indices] - peak_time) / decay_time) * \
                   np.cos(2 * np.pi * frequency * (t[falling_indices] - peak_time) + phase)
    rising_indices = np.where(t < peak_time)
    qpo[rising_indices] = amplitude * np.exp(+(t[rising_indices] - peak_time) / decay_time) * \
                   np.cos(2 * np.pi * frequency * (t[rising_indices] - peak_time) + phase)
    return qpo


def sine_model(times, amplitude, frequency, phase, **kwargs):
    t = deepcopy(times)
    t -= times[0]
    return amplitude * np.sin(2*np.pi*t*frequency + phase)


def sine_gaussian(t, mu, sigma, amplitude, frequency, phase, **kwargs):
    return sine_model(times=t, amplitude=amplitude, frequency=frequency, phase=phase) \
           * norm_gaussian(t=t, t_0=mu, sigma=sigma)


def exponential_background(times, tau, offset, **kwargs):
    return np.exp(times/tau) + offset


def polynomial(t, a0, a1, a2, a3, a4):
    times = t.copy()
    times -= times[0]
    times -= times[-1] / 2
    return a0 + a1 * times + a2 * times**2 + a3 * times**3 + a4 * times**4


def gaussian(t, amplitude, t_0, sigma):
    return amplitude * np.exp(-(t - t_0) ** 2 / (2 * sigma ** 2))


def norm_gaussian(t, t_0, sigma, **kwargs):
    amplitude = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    return gaussian(t=t, amplitude=amplitude, t_0=t_0, sigma=sigma)


def log_normal(t, amplitude, t_0, sigma):
    return amplitude / t * np.exp(-(np.log(t) - t_0)**2 / (2 * sigma**2))


def lorentzian(t, amplitude, t_0, sigma):
    return amplitude * (sigma / ((t - t_0) ** 2 + (sigma ** 2)))


def stabilised_burst_envelope(t, amplitude, t_max, sigma, skewness):
    return bar_lev(burst_envelope(t, amplitude=amplitude, t_max=t_max, sigma=sigma, skewness=skewness))


def stabilised_exponential(t, tau, offset):
    return bar_lev(exponential_background(t, tau=tau, offset=offset))
