import numpy as np
import stingray

from QPOEstimation.model.series import fast_rise_exponential_decay


def red_noise(frequencies, alpha, beta):
    return beta * frequencies ** (-alpha)


def white_noise(frequencies, sigma):
    return sigma * np.ones(len(frequencies))


def broken_power_law_noise(frequencies, alpha_1, alpha_2, beta, delta, rho):
    return beta * frequencies ** (-alpha_1) * (1 + (frequencies / delta) ** ((alpha_2 - alpha_1) / rho)) ** (-rho)


def lorentzian(frequencies, amplitude, central_frequency, width, offset):
    return amplitude * (width ** 2 / ((frequencies - central_frequency) ** 2 + width ** 2)) / np.pi / width + offset


def burst_envelope_ps(frequencies, amplitude, t_start, t_max, sigma, skewness):
    delta_freq = frequencies[1] - frequencies[0]
    duration = 1 / delta_freq
    times = np.arange(t_start, t_start + duration, 1 / int(frequencies[-1] * duration))
    envelope = fast_rise_exponential_decay(times=times, amplitude=amplitude, t_0=t_max, sigma=sigma, skewness=skewness)
    lc = stingray.Lightcurve.make_lightcurve(envelope, dt=times[1] - times[0])
    ps = stingray.Powerspectrum(lc)
    return ps