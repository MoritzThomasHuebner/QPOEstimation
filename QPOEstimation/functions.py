import numpy as np

import bilby
import stingray

from QPOEstimation.prior import SlabSpikePrior


def red_noise(frequencies, alpha, beta):
    return beta * frequencies ** (-alpha)


def white_noise(frequencies, sigma):
    return sigma * np.ones(len(frequencies))


def broken_power_law_noise(frequencies, alpha_1, alpha_2, beta, delta, rho):
    return beta * frequencies ** (-alpha_1) * (1 + (frequencies / delta) ** ((alpha_2 - alpha_1) / rho)) ** (-rho)


def lorentzian(frequencies, amplitude, central_frequency, width, offset):
    return amplitude * (width ** 2 / ((frequencies - central_frequency) ** 2 + width ** 2)) / np.pi / width + offset


def qpo_shot(times, offset, amplitude, frequency, t_qpo, phase, decay_time):
    if amplitude == 0:
        return np.zeros(len(times))
    res = np.zeros(len(times))
    idxs = [np.where(times >= t_qpo)][0]
    res[idxs] = 0.5*amplitude*(1 + np.cos(2 * np.pi * frequency * (times[idxs] - t_qpo) + phase))/2 * np.exp(
        -(times[idxs] - t_qpo) / decay_time)
    return res


def gaussian(x, mu, sigma):
    return np.exp(-(x - mu) ** 2. / (2 * sigma ** 2.)) / np.sqrt(2 * np.pi * sigma ** 2)


def burst_envelope(times, amplitude, t_max, sigma, skewness):
    if amplitude == 0:
        return np.zeros(len(times))
    before_burst_indices = np.where(times <= t_max)
    after_burst_indices = np.where(times > t_max)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = amplitude * np.exp((times[before_burst_indices] - t_max) / sigma)
    envelope[after_burst_indices] = amplitude * np.exp(-(times[after_burst_indices] - t_max) / sigma / skewness)
    return envelope


def burst_envelope_ps(frequencies, amplitude, t_start, t_max, sigma, skewness):
    delta_freq = frequencies[1] - frequencies[0]
    duration = 1 / delta_freq
    times = np.arange(t_start, t_start + duration, 1 / int(frequencies[-1] * duration))
    envelope = burst_envelope(times=times, amplitude=amplitude, t_max=t_max, sigma=sigma, skewness=skewness)
    lc = stingray.Lightcurve.make_lightcurve(envelope, dt=times[1] - times[0])
    ps = stingray.Powerspectrum(lc)
    return ps


def burst_qpo_model(times, background_rate=0,
                    amplitude_0=0, t_max_0=0, sigma_0=1, skewness_0=1,
                    amplitude_1=0, t_max_1=0, sigma_1=1, skewness_1=1,
                    amplitude_2=0, t_max_2=0, sigma_2=1, skewness_2=1,
                    amplitude_3=0, t_max_3=0, sigma_3=1, skewness_3=1,
                    amplitude_4=0, t_max_4=0, sigma_4=1, skewness_4=1,
                    amplitude_qpo_0=0, phase_0=0, frequency_0=1, t_qpo_0=0, decay_time_0=1,
                    amplitude_qpo_1=0, phase_1=0, frequency_1=1, t_qpo_1=0, decay_time_1=1,
                    **kwargs):
    offset_0 = amplitude_qpo_0
    offset_1 = amplitude_qpo_1
    return \
        burst_envelope(times=times, amplitude=amplitude_0, t_max=t_max_0, sigma=sigma_0, skewness=skewness_0) + \
        burst_envelope(times=times, amplitude=amplitude_1, t_max=t_max_1, sigma=sigma_1, skewness=skewness_1) + \
        burst_envelope(times=times, amplitude=amplitude_2, t_max=t_max_2, sigma=sigma_2, skewness=skewness_2) + \
        burst_envelope(times=times, amplitude=amplitude_3, t_max=t_max_3, sigma=sigma_3, skewness=skewness_3) + \
        burst_envelope(times=times, amplitude=amplitude_4, t_max=t_max_4, sigma=sigma_4, skewness=skewness_4) + \
        qpo_shot(times=times, offset=offset_0, amplitude=amplitude_qpo_0, frequency=frequency_0, t_qpo=t_qpo_0,
                 phase=phase_0, decay_time=decay_time_0) + \
        qpo_shot(times=times, offset=offset_1, amplitude=amplitude_qpo_1, frequency=frequency_1, t_qpo=t_qpo_1,
                 phase=phase_1, decay_time=decay_time_1) + \
        background_rate


def burst_qpo_model_norm(times, background_rate=0,
                         amplitude_0=0, t_max_0=0, sigma_0=1, skewness_0=1,
                         amplitude_1=0, t_max_1=0, sigma_1=1, skewness_1=1,
                         amplitude_2=0, t_max_2=0, sigma_2=1, skewness_2=1,
                         amplitude_3=0, t_max_3=0, sigma_3=1, skewness_3=1,
                         amplitude_4=0, t_max_4=0, sigma_4=1, skewness_4=1,
                         amplitude_qpo_0=0, phase_0=0, frequency_0=1, t_qpo_0=0, decay_time_0=1,
                         amplitude_qpo_1=0, phase_1=0, frequency_1=1, t_qpo_1=0, decay_time_1=1,
                         **kwargs):
    T = times[-1] - times[0]
    nbin = len(times)
    norm = nbin/T
    return burst_qpo_model(times, background_rate,
                           amplitude_0, t_max_0, sigma_0, skewness_0,
                           amplitude_1, t_max_1, sigma_1, skewness_1,
                           amplitude_2, t_max_2, sigma_2, skewness_2,
                           amplitude_3, t_max_3, sigma_3, skewness_3,
                           amplitude_4, t_max_4, sigma_4, skewness_4,
                           amplitude_qpo_0, phase_0, frequency_0, t_qpo_0, decay_time_0,
                           amplitude_qpo_1, phase_1, frequency_1, t_qpo_1, decay_time_1,
                           **kwargs) / norm
