import numpy as np


def polynomial(t, a0, a1, a2, a3, a4):
    times = t.copy()
    times -= times[0]
    times -= times[-1] / 2
    return a0 + a1 * times + a2 * times**2 + a3 * times**3 + a4 * times**4


def fred(times, amplitude, t_0, sigma_rise, sigma_fall):
    if amplitude == 0:
        return np.zeros(len(times))
    before_burst_indices = np.where(times <= t_0)
    after_burst_indices = np.where(times > t_0)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = amplitude * np.exp((times[before_burst_indices] - t_0) / sigma_rise)
    envelope[after_burst_indices] = amplitude * np.exp(-(times[after_burst_indices] - t_0) / sigma_fall)
    return envelope


def fred_norris(times, amplitude, sigma, t_0, tau):
    frac = (times-t_0)/tau
    return amplitude * np.exp(-sigma*(frac + 1/frac))


def fred_norris_extended(times, amplitude, sigma, t_0, tau, gamma, nu):
    frac = (times-t_0)/tau
    return amplitude * np.exp(-sigma**gamma * frac**nu -sigma**gamma * 1/frac**nu)


def exponential_background(times, amplitude, tau, offset, **kwargs):
    return amplitude * np.exp(times/tau) + offset


def gaussian(t, amplitude, t_0, sigma):
    return amplitude * np.exp(-(t - t_0) ** 2 / (2 * sigma ** 2))


def skew_gaussian(times, amplitude, t_0, sigma_rise, sigma_fall):
    before_burst_indices = np.where(times <= t_0)
    after_burst_indices = np.where(times > t_0)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = gaussian(t=times[before_burst_indices], amplitude=amplitude, t_0=t_0, sigma=sigma_rise)
    envelope[after_burst_indices] = gaussian(t=times[after_burst_indices], amplitude=amplitude, t_0=t_0, sigma=sigma_fall)
    return envelope


def log_normal(t, amplitude, t_0, sigma):
    return amplitude / t * np.exp(-(np.log(t) - t_0)**2 / (2 * sigma**2))


def lorentzian(t, amplitude, t_0, sigma):
    return amplitude * (sigma / ((t - t_0) ** 2 + (sigma ** 2)))
