import numpy as np


def polynomial(times, a0, a1, a2, a3, a4):
    t = times.copy()
    t -= t[0]
    t -= t[-1] / 2
    return a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4


def fred(times, log_amplitude, t_0, log_sigma_rise, log_sigma_fall):
    amplitude = np.exp(log_amplitude)
    sigma_rise = np.exp(log_sigma_rise)
    sigma_fall = np.exp(log_sigma_fall)

    before_burst_indices = np.where(times <= t_0)
    after_burst_indices = np.where(times > t_0)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = amplitude * np.exp((times[before_burst_indices] - t_0) / sigma_rise)
    envelope[after_burst_indices] = amplitude * np.exp(-(times[after_burst_indices] - t_0) / sigma_fall)
    return envelope


def fred_norris(times, log_amplitude, log_psi, t_0, delta):
    amplitude = np.exp(log_amplitude)
    psi = np.exp(log_psi)

    frac = (times + delta) / t_0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return amplitude * np.exp(-psi * (frac + 1 / frac)) * np.exp(2 * psi)


def fred_norris_extended(times, log_amplitude, log_psi, t_0, delta, log_gamma, log_nu):
    amplitude = np.exp(log_amplitude)
    nu = np.exp(log_nu)
    gamma = np.exp(log_gamma)
    psi = np.exp(log_psi)

    frac = (times + delta) / t_0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return amplitude * np.exp(-psi**gamma * frac**gamma - psi**nu / frac**nu) * np.exp(2 * psi)


def gaussian(times, log_amplitude, t_0, log_sigma):
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


def piecewise_linear(times, beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, k_2, k_3, k_4, k_5):
    duration = times[-1] - times[0]
    times = 2 * times / duration - 1
    betas = np.array([beta_1, beta_2, beta_3, beta_4, beta_5])
    ks = 2*np.array([0, k_2, k_3, k_4, k_5])/duration - 1
    return beta_0 + np.sum([_linear(times, betas[i], ks[i]) for i in range(len(betas))], axis=0)


def _linear(times, beta, k):
    res = beta * (times - k)
    res[np.where(times < k)] = 0
    return res


def piecewise_cubic(times, beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, k_4, k_5):
    duration = times[-1] - times[0]
    times = 2 * times / duration - 1
    betas = np.array([beta_3, beta_4, beta_5])
    ks = 2 * np.array([0, k_4, k_5]) / duration - 1
    return beta_0 + beta_1*times + beta_2 * times**2 + np.sum([_cubic(times, betas[i], ks[i]) for i in range(len(betas))], axis=0)


def _cubic(times, beta, k):
    res = beta * (times - k)**3
    res[np.where(times < k)] = 0
    return res
