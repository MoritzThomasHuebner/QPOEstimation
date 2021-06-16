import bilby.gw.detector.psd
import numpy as np
import QPOEstimation
from scipy.signal import periodogram

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")


frequencies = np.linspace(1/100000, 20, 1000000)
alpha = 2
beta = 50
# white_noise = 0.0000001
white_noise = 4 #good parameter
amplitude = 50
width = 0.1
central_frequency = 1
sampling_frequency = 40
duration_signal = 40
duration_white_noise = 40
x_break = beta/white_noise * central_frequency**(-alpha)
print(x_break)


psd_array_noise = QPOEstimation.model.psd.red_noise(frequencies=frequencies, alpha=alpha, beta=beta) + white_noise
psd_array_qpo = QPOEstimation.model.psd.lorentzian(frequencies=frequencies, amplitude=amplitude, width=width, central_frequency=central_frequency)
psd_array_white_noise = white_noise * np.ones(len(frequencies))

psd_noise = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
    frequency_array=frequencies, psd_array=psd_array_noise)
psd_signal = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
    frequency_array=frequencies, psd_array=psd_array_noise + psd_array_qpo)
psd_white_noise = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
    frequency_array=frequencies, psd_array=psd_array_white_noise)
psd_qpo = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
    frequency_array=frequencies, psd_array=psd_array_qpo)



series_signal = bilby.core.series.CoupledTimeAndFrequencySeries(sampling_frequency=sampling_frequency, duration=duration_signal)
series_white_noise = bilby.core.series.CoupledTimeAndFrequencySeries(duration=duration_white_noise, sampling_frequency=sampling_frequency)
fd_data_signal, _ = psd_signal.get_noise_realisation(sampling_frequency=sampling_frequency, duration=duration_signal)
td_data_signal = bilby.core.utils.infft(frequency_domain_strain=fd_data_signal, sampling_frequency=sampling_frequency)
fd_data_white_noise, freqs_white_noise = psd_white_noise.get_noise_realisation(sampling_frequency=sampling_frequency, duration=duration_white_noise)
td_data_white_noise = bilby.core.utils.infft(frequency_domain_strain=fd_data_white_noise, sampling_frequency=sampling_frequency)

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

plt.plot(series_signal.time_array, td_data_signal)
plt.show()


plt.plot(series_signal.time_array, td_data_signal)
plt.plot(series_white_noise.time_array, td_data_white_noise)
plt.show()


# freqs_test, powers_test = periodogram(td_data_signal, fs=sampling_frequency, window='hann')
# plt.loglog()

combined_series = bilby.core.series.CoupledTimeAndFrequencySeries(duration=duration_signal + duration_white_noise, sampling_frequency=sampling_frequency)
combined_signal = np.append(np.split(td_data_white_noise, 2)[0], np.append(td_data_signal, np.split(td_data_white_noise, 2)[1]))
plt.plot(combined_series.time_array, combined_signal)
plt.show()

freqs_combined_periodogram, powers_combined_periodogram = periodogram(combined_signal, fs=sampling_frequency, window='hann')

plt.loglog(freqs_signal_periodogram, powers_signal_periodogram)
plt.loglog(freqs_combined_periodogram, powers_combined_periodogram)
plt.show()
assert False
# times_save = combined_series.time_array - 5000
# idx_1 = QPOEstimation.utils.get_indices_by_time(times_save, minimum_time=-6000, maximum_time=-20)
# idx_2 = QPOEstimation.utils.get_indices_by_time(times_save, minimum_time=20, maximum_time=6000)
# combined_signal[idx_1] = 0
# combined_signal[idx_2] = 0
#
# plt.plot(times_save, combined_signal)
# plt.show()
# res = np.array([times_save, combined_signal]).T
# np.savetxt('injection_files_pop/general_qpo/whittle/00_data.txt', res)
#
# import json
# params = dict(amplitude=amplitude, alpha=alpha, beta=beta, central_frequency=central_frequency, width=width, sigma=white_noise)
#
# with open('injection_files_pop/general_qpo/whittle/00_params.json', 'w') as f:
#     json.dump(params, f)


snr_qpo_optimals = []
durations_white_noise = np.arange(0, 10000, 50)
extension_factors = (durations_white_noise + duration_signal) / duration_signal


for duration_white_noise, extension_factor in zip(durations_white_noise, extension_factors):
    if duration_white_noise != 0:
        fd_data_white_noise, freqs_white_noise = psd_white_noise.get_noise_realisation(
            sampling_frequency=sampling_frequency, duration=duration_white_noise)
        td_data_white_noise = bilby.core.utils.infft(frequency_domain_strain=fd_data_white_noise,
                                                 sampling_frequency=sampling_frequency)
    else:
        td_data_white_noise = np.array([])
    combined_signal = np.append(np.split(td_data_white_noise, 2)[0],
                                np.append(td_data_signal, np.split(td_data_white_noise, 2)[1]))
    freqs_combined_periodogram, powers_combined_periodogram = periodogram(combined_signal, fs=sampling_frequency,
                                                                          window='hann')
    psd_array_noise_diluted = QPOEstimation.model.psd.red_noise(frequencies=frequencies, alpha=alpha,
                                                                beta=beta) / extension_factor + white_noise
    psd_array_qpo_diluted = psd_array_qpo / extension_factor

    psd_noise_diluted = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
        frequency_array=frequencies, psd_array=psd_array_noise_diluted)
    psd_qpo_diluted = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
        frequency_array=frequencies, psd_array=psd_array_qpo_diluted)
    snr_qpo_optimal = np.sqrt(np.sum(
        np.nan_to_num((psd_qpo_diluted.power_spectral_density_interpolated(freqs_combined_periodogram) /
                       psd_noise_diluted.power_spectral_density_interpolated(freqs_combined_periodogram)) ** 2, nan=0)))
    print(snr_qpo_optimal)
    print(extension_factor)
    print()
    snr_qpo_optimals.append(snr_qpo_optimal)

plt.plot(extension_factors, snr_qpo_optimals)
# plt.axvline(x_break, label='$x_{\mathrm{break}}$', color='black', linestyle='-.')
plt.xlabel(r'Extension factor $x$')
plt.ylabel(r'SNR')
plt.legend()
plt.savefig('periodogram_pop/snr_vs_extension_zeros.pdf')
plt.show()