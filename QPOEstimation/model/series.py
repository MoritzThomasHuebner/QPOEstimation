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


def fred_norris(times, amplitude, log_psi, t_0, delta):
    psi = np.exp(log_psi)
    frac = (times + delta) / t_0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return amplitude * np.exp(-psi * (frac + 1 / frac)) * np.exp(2 * psi)


def fred_norris_extended(times, amplitude, log_psi, t_0, delta, log_gamma, log_nu):
    nu = np.exp(log_nu)
    gamma = np.exp(log_gamma)
    psi = np.exp(log_psi)
    frac = (times + delta) / t_0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return amplitude * np.exp(-psi**gamma * frac**gamma - psi**nu /frac**nu) * np.exp(2 * psi)



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
