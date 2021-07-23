import json
import bilby.gw.detector.psd
import numpy as np
import scipy.signal.windows

import QPOEstimation
from scipy.signal import periodogram

import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d

matplotlib.use("Qt5Agg")


# 06: Red noise + white noise + 40s QPO Zeros extended, no trend
# 07: Red noise + white noise + 40s QPO White noise extended, no trend
# 08: White noise + non-stationary 20s QPO + Exponential pulse profile
# 09: QPO, zeros extended, no trend
# 10: QPO, white noise extended, no trend
# 11: Poissonian, 20s QPO, white noise extended. Gaussian trend.

frequencies = np.linspace(1/100000, 20, 1000000)
alpha = 2
beta = 0
white_noise = 0
amplitude = 3000
width = 0.1
central_frequency = 5
sampling_frequency = 40
duration_signal = 20
duration_white_noise = 400
# x_break = beta/white_noise * central_frequency**(-alpha)
# print(x_break)
extension_modes = ['zeros', 'white_noise']
extension_mode = extension_modes[0]
injection_id = "11"
if injection_id == "11":
    profile_amplitude = 1000
    profile_t_0 = 200
    sigma = 20
    background_rate = 10
elif injection_id == "08":
    profile_amplitude = 5000
    profile_t_0 = 70
    sigma_fall = 20
    sigma_rise = 10

# Only add white noise when we extend with zeros
if extension_mode == extension_modes[0]:
    psd_array_noise = QPOEstimation.model.psd.red_noise(frequencies=frequencies, alpha=alpha, beta=beta) + white_noise
elif extension_mode == extension_modes[1]:
    psd_array_noise = QPOEstimation.model.psd.red_noise(frequencies=frequencies, alpha=alpha, beta=beta)
    psd_array_white_noise = white_noise * np.ones(len(frequencies))
    psd_white_noise = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
        frequency_array=frequencies, psd_array=psd_array_white_noise)
else:
    raise ValueError

psd_array_qpo = QPOEstimation.model.psd.lorentzian(
    frequencies=frequencies, amplitude=amplitude, width=width, central_frequency=central_frequency)

psd_noise = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
    frequency_array=frequencies, psd_array=psd_array_noise)
psd_signal = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
    frequency_array=frequencies, psd_array=psd_array_noise + psd_array_qpo)
psd_qpo = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
    frequency_array=frequencies, psd_array=psd_array_qpo)


series_signal = bilby.core.series.CoupledTimeAndFrequencySeries(sampling_frequency=sampling_frequency, duration=duration_signal)
series_combined = bilby.core.series.CoupledTimeAndFrequencySeries(duration=duration_white_noise, sampling_frequency=sampling_frequency)
fd_data_signal, _ = psd_signal.get_noise_realisation(sampling_frequency=sampling_frequency, duration=duration_signal)
td_data_signal = bilby.core.utils.infft(frequency_domain_strain=fd_data_signal, sampling_frequency=sampling_frequency)
if extension_mode == extension_modes[0]:
    td_data_extension = np.zeros(len(series_combined.time_array))
elif extension_mode == extension_modes[1]:
    fd_data_white_noise, freqs_white_noise = psd_white_noise.get_noise_realisation(
        sampling_frequency=sampling_frequency, duration=duration_white_noise)
    td_data_extension = bilby.core.utils.infft(
        frequency_domain_strain=fd_data_white_noise, sampling_frequency=sampling_frequency)
else:
    raise ValueError

freqs_signal_periodogram, powers_signal_periodogram = periodogram(td_data_signal, fs=sampling_frequency, window='hann')

snr_qpo_optimal = np.sqrt(np.sum(
    np.nan_to_num((psd_qpo.power_spectral_density_interpolated(freqs_signal_periodogram) /
                   psd_noise.power_spectral_density_interpolated(freqs_signal_periodogram)) ** 2, nan=0)))


print(snr_qpo_optimal)


plt.loglog(frequencies, psd_array_noise)
plt.loglog(frequencies, psd_array_qpo)
plt.loglog(series_signal.frequency_array, np.abs(fd_data_signal) / np.sqrt(series_signal.frequency_array))
# plt.loglog(freqs_white_noise, np.abs(fd_data_white_noise)/np.sqrt(freqs_white_noise))
plt.loglog(freqs_signal_periodogram, powers_signal_periodogram)
plt.show()

td_data_signal_windowed = td_data_signal * scipy.signal.windows.hann(len(td_data_signal))

plt.plot(series_signal.time_array, td_data_signal)
plt.plot(series_signal.time_array, td_data_signal_windowed)
plt.show()


signal_indices = QPOEstimation.utils.get_indices_by_time(
    minimum_time=duration_white_noise/2 - duration_signal/2 - 0.001,
    maximum_time=duration_white_noise/2 + duration_signal/2, times=series_combined.time_array)
combined_signal = td_data_extension
combined_signal[signal_indices] += td_data_signal_windowed

if injection_id == "08":
    combined_signal += QPOEstimation.model.mean.fred(
        times=series_combined.time_array, log_amplitude=np.log(profile_amplitude), t_0=profile_t_0,
        log_sigma_fall=np.log(sigma_fall), log_sigma_rise=np.log(sigma_rise))

if injection_id == "11":
    combined_signal += QPOEstimation.model.mean.gaussian(
        times=series_combined.time_array, log_amplitude=np.log(profile_amplitude), t_0=profile_t_0,
        log_sigma=np.log(sigma)) + background_rate
    interp_func = interp1d(x=series_combined.time_array, y=combined_signal)
    combined_signal = QPOEstimation.poisson.poisson_process(times=series_combined.time_array, func=interp_func)

plt.plot(series_combined.time_array, combined_signal)
plt.show()

freqs_combined_periodogram, powers_combined_periodogram = periodogram(combined_signal, fs=sampling_frequency, window='hann')

plt.loglog(freqs_signal_periodogram, powers_signal_periodogram)
plt.loglog(freqs_combined_periodogram, powers_combined_periodogram)
plt.xlim(1/sampling_frequency, sampling_frequency/2)
plt.show()


times_save = series_combined.time_array - duration_white_noise/2
plt.plot(times_save, combined_signal)
plt.show()

np.savetxt(f'injection_files_pop/general_qpo/whittle/{injection_id}_data.txt', np.array([times_save, combined_signal]).T)

params = dict(amplitude=amplitude, alpha=alpha, beta=beta, central_frequency=central_frequency, width=width, sigma=white_noise)

with open(f'injection_files_pop/general_qpo/whittle/{injection_id}_params.json', 'w') as f:
    json.dump(params, f)
