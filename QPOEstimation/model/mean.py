import numpy as np


def polynomial(times, a0, a1, a2, a3, a4):
    t = times.copy()
    t -= t[0]
    t -= t[-1] / 2
    return a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4


def skew_exponential(times, log_amplitude, t_0, log_sigma_rise, log_sigma_fall):
    amplitude = np.exp(log_amplitude)
    sigma_rise = np.exp(log_sigma_rise)
    sigma_fall = np.exp(log_sigma_fall)

    before_burst_indices = np.where(times <= t_0)
    after_burst_indices = np.where(times > t_0)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = amplitude * np.exp((times[before_burst_indices] - t_0) / sigma_rise)
    envelope[after_burst_indices] = amplitude * np.exp(-(times[after_burst_indices] - t_0) / sigma_fall)
    return envelope


def fred(times, log_amplitude, log_psi, t_0, delta):
    amplitude = np.exp(log_amplitude)
    psi = np.exp(log_psi)

    frac = (times + delta) / t_0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return amplitude * np.exp(-psi * (frac + 1 / frac)) * np.exp(2 * psi)


def fred_extended(times, log_amplitude, log_psi, t_0, delta, log_gamma, log_nu):
    amplitude = np.exp(log_amplitude)
    nu = np.exp(log_nu)
    gamma = np.exp(log_gamma)
    psi = np.exp(log_psi)

    frac = (times + delta) / t_0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return amplitude * np.exp(-psi**gamma * frac**gamma - psi**nu / frac**nu) * np.exp(2 * psi)


def gaussian(times, log_amplitude, t_0, log_sigma, **kwargs):
    amplitude = np.exp(log_amplitude)
    sigma = np.exp(log_sigma)
    return amplitude * np.exp(-(times - t_0) ** 2 / (2 * sigma ** 2))


def skew_gaussian(times, log_amplitude, t_0, log_sigma_rise, log_sigma_fall):
    before_burst_indices = np.where(times <= t_0)
    after_burst_indices = np.where(times > t_0)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = gaussian(times=times[before_burst_indices], log_amplitude=log_amplitude,
                                              t_0=t_0, log_sigma=log_sigma_rise)
    envelope[after_burst_indices] = gaussian(times=times[after_burst_indices], log_amplitude=log_amplitude,
                                             t_0=t_0, log_sigma=log_sigma_fall)
    return envelope


def log_normal(times, log_amplitude, t_0, log_sigma):
    amplitude = np.exp(log_amplitude)
    sigma = np.exp(log_sigma)
    return amplitude / times * np.exp(-(np.log(times) - t_0) ** 2 / (2 * sigma ** 2))


def lorentzian(times, log_amplitude, t_0, log_sigma):
    amplitude = np.exp(log_amplitude)
    sigma = np.exp(log_sigma)
    return amplitude * (sigma / ((times - t_0) ** 2 + (sigma ** 2)))
