import numpy as np
import stingray


def red_noise(frequencies, alpha, beta):
    return beta * frequencies ** (-alpha)


def white_noise(frequencies, sigma):
    return sigma * np.ones(len(frequencies))


def broken_power_law_noise(frequencies, alpha_1, alpha_2, beta, delta, rho):
    return beta*frequencies**(-alpha_1) * (1 + (frequencies/delta)**((alpha_2-alpha_1)/rho))**(-rho)


def lorentzian(frequencies, amplitude, central_frequency, width, offset):
    return amplitude * (width ** 2 / ((frequencies - central_frequency) ** 2 + width ** 2)) / np.pi / width + offset


def qpo_shot(times, offset, amplitude, frequency, t_0, phase, decay_time):
    res = np.zeros(len(times))
    idxs = [np.where(times >= t_0)][0]
    res[idxs] = (offset + amplitude*np.cos(2*np.pi*frequency*(times[idxs] - t_0) + phase)) * np.exp(-(times[idxs] - t_0)/decay_time)
    return res


def gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2. / (2 * sigma**2.)) / np.sqrt(2 * np.pi * sigma**2)


def burst_envelope(times, amplitude, t_max, sigma, skewness):
    before_burst_indices = np.where(times <= t_max)
    after_burst_indices = np.where(times > t_max)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = amplitude * np.exp((times[before_burst_indices] - t_max) / sigma)
    envelope[after_burst_indices] = amplitude * np.exp(-(times[after_burst_indices] - t_max) / sigma / skewness)
    return envelope


def burst_envelope_ps(frequencies, amplitude, t_start, t_max, sigma, skewness):
    delta_freq = frequencies[1] - frequencies[0]
    duration = 1 / delta_freq
    times = np.arange(t_start, t_start + duration, 1/int(frequencies[-1] * duration))
    envelope = burst_envelope(times=times, amplitude=amplitude, t_max=t_max, sigma=sigma, skewness=skewness)
    lc = stingray.Lightcurve.make_lightcurve(envelope, dt=times[1] - times[0])
    ps = stingray.Powerspectrum(lc)
    return ps