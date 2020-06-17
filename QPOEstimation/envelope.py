import numpy as np

import bilby
import stingray


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
    print(times)
    print(envelope)
    lc = stingray.Lightcurve.make_lightcurve(envelope, dt=times[1] - times[0])
    ps = stingray.Powerspectrum(lc)
    return ps
