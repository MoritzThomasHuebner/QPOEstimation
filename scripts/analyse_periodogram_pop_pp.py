import os
from pathlib import Path

import bilby.core.series
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import periodogram

import QPOEstimation
from QPOEstimation.prior.gp import *
from pathlib import Path
import sys
import json


matplotlib.use('Qt5Agg')

modes = ['zeros', 'white_noise']
mode = 0


end_times = np.arange(20, 200, 20)
start_times = -end_times
durations = 2 * end_times


outdir_qpo_periodogram = f'injection/general_qpo_injection/general_qpo_recovery/whittle/results/'
outdir_red_noise_periodogram = f'injection/general_qpo_injection/red_noise_recovery/whittle/results/'

log_frequency_spreads_zero_padded = []
injection_id = str(mode).zfill(2)
data = np.loadtxt(f'injection_files_pop/general_qpo/whittle/{injection_id}_data.txt')
times = data[:, 0]
y = data[:, 1]
sampling_frequency = int(round(1/(times[1] - times[0])))

with open(f'injection_files_pop/general_qpo/whittle/{injection_id}_params.json', 'r') as f:
    params = json.load(f)

frequencies = np.linspace(1/100000, 20, 1000000)
alpha = params['alpha']
beta = params['beta']
white_noise = params['sigma']
amplitude = params['amplitude']
width = params['width']
central_frequency = params['central_frequency']

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

duration_signal = 40
duration_white_noise = 9960
x_break = beta/white_noise * central_frequency**(-alpha)
print(x_break)

ln_bfs = []
log_frequency_spreads = []
ln_bfs_zero_padded = []
durations_reduced = []
snr_optimals = []
snr_matched_filters = []
extension_factors = []

for start_time, end_time, duration in zip(start_times, end_times, durations):
    extension_factor = duration/durations[0]
    duration_white_noise = duration - durations[0]
    series = bilby.core.series.CoupledTimeAndFrequencySeries(duration=duration, sampling_frequency=sampling_frequency, start_time=start_time)
    idxs = QPOEstimation.utils.get_indices_by_time(times=times, minimum_time=start_time, maximum_time=end_time)
    y_selected = y[idxs]

    label_red_noise = f'{injection_id}_{float(start_time)}_{float(end_time)}'
    label_qpo = f'{injection_id}_{float(start_time)}_{float(end_time)}'
    try:
        res_red_noise_periodogram = bilby.result.read_in_result(outdir=outdir_red_noise_periodogram, label=label_red_noise)
        res_qpo_periodogram = bilby.result.read_in_result(outdir=outdir_qpo_periodogram, label=label_qpo)
        ln_bfs.append(res_qpo_periodogram.log_evidence - res_red_noise_periodogram.log_evidence)
        log_frequency_spreads.append(np.std(res_qpo_periodogram.posterior['log_frequency']))
        # plt.hist(np.exp(res_qpo_periodogram.posterior['log_frequency']), histtype='step', label=f"{duration}s",
        #          density=True, bins='fd')
        durations_reduced.append(duration)
        extension_factors.append(extension_factor)
    except Exception as e:
        print(e)
        continue
    if duration_white_noise != 0:
        fd_data_white_noise, freqs_white_noise = psd_white_noise.get_noise_realisation(
            sampling_frequency=sampling_frequency, duration=duration_white_noise)
        td_data_white_noise = bilby.core.utils.infft(frequency_domain_strain=fd_data_white_noise,
                                                 sampling_frequency=sampling_frequency)
    else:
        td_data_white_noise = np.array([])

    ### Optimal SNR calculation
    freqs_combined_periodogram, powers_combined_periodogram = \
        periodogram(y_selected, fs=sampling_frequency, window='hann')
    psd_array_noise_diluted = QPOEstimation.model.psd.red_noise(
        frequencies=frequencies, alpha=alpha,
        beta=beta) / extension_factor + white_noise
    psd_array_qpo_diluted = psd_array_qpo / extension_factor
    psd_array_signal_diluted = psd_array_noise_diluted + psd_array_qpo_diluted
    psd_noise_diluted = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
        frequency_array=frequencies, psd_array=psd_array_noise_diluted)
    psd_qpo_diluted = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
        frequency_array=frequencies, psd_array=psd_array_qpo_diluted)
    psd_signal_diluted = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
        frequency_array=frequencies, psd_array=psd_array_signal_diluted)
    snr_qpo_optimal = np.sqrt(np.sum(
        np.nan_to_num((psd_qpo_diluted.power_spectral_density_interpolated(freqs_combined_periodogram) /
                       psd_noise_diluted.power_spectral_density_interpolated(freqs_combined_periodogram)) ** 2, nan=0)))
    print(snr_qpo_optimal)
    snr_optimals.append(snr_qpo_optimal)

    ### Matched filter SNR calculation, doesnt make sense yet
    max_like_params = res_qpo_periodogram.posterior.iloc[-1]
    alpha_max_like = max_like_params['alpha']
    beta_max_like = np.exp(max_like_params['log_beta'])
    white_noise_max_like = np.exp(max_like_params['log_sigma'])
    amplitude_max_like = np.exp(max_like_params['log_amplitude'])
    width_max_like = np.exp(max_like_params['log_width'])
    central_frequency_max_like = np.exp(max_like_params['log_frequency'])
    # psd_array_noise_max_like = QPOEstimation.model.psd.red_noise(frequencies=freqs_combined_periodogram, alpha=alpha_max_like, beta=beta_max_like) + white_noise_max_like
    # psd_array_qpo_max_like = QPOEstimation.model.psd.lorentzian(
    #     frequencies=freqs_combined_periodogram, amplitude=amplitude_max_like, width=width_max_like,
    #     central_frequency=central_frequency_max_like)
    # powers_combined_periodogram_qpo_divided = powers_combined_periodogram / psd_array_qpo_max_like
    # snr_qpo_matched_filter = np.sqrt(np.sum(
    #     np.nan_to_num((psd_array_qpo_max_like / powers_combined_periodogram_qpo_divided) ** 2, nan=0)))
    # print(snr_qpo_matched_filter)
    # snr_matched_filters.append(snr_qpo_matched_filter)

    # plt.loglog(freqs_combined_periodogram, powers_combined_periodogram)
    # plt.loglog(freqs_combined_periodogram, powers_combined_periodogram_qpo_divided)
    # plt.loglog(frequencies, psd_signal_diluted.psd_array)
    # plt.loglog(freqs_combined_periodogram, psd_array_qpo_max_like)
    # plt.loglog(freqs_combined_periodogram, psd_array_noise_max_like)
    # plt.show()

    ### Chi-squared tests
    chi_square = QPOEstimation.model.psd.periodogram_chi_square_test(
        frequencies=freqs_combined_periodogram, powers=powers_combined_periodogram,
        psd=psd_signal_diluted, degrees_of_freedom=6)
    print(chi_square)

    idxs = QPOEstimation.utils.get_indices_by_time(freqs_combined_periodogram,
                                                   minimum_time=central_frequency_max_like-width_max_like,
                                                   maximum_time=central_frequency_max_like+width_max_like)
    chi_square = QPOEstimation.model.psd.periodogram_chi_square_test(
        frequencies=freqs_combined_periodogram[idxs], powers=powers_combined_periodogram[idxs],
        psd=psd_signal_diluted, degrees_of_freedom=6)
    print(chi_square)
    assert False




np.savetxt('periodogram_pop/temp_ln_bfs', ln_bfs)
np.savetxt('periodogram_pop/temp_log_frequency_spreads', log_frequency_spreads)
np.savetxt('periodogram_pop/temp_ln_bfs_zero_padded', ln_bfs_zero_padded)
np.savetxt('periodogram_pop/temp_durations_reduced', durations_reduced)
np.savetxt('periodogram_pop/temp_snr_optimals', snr_optimals)
np.savetxt('periodogram_pop/temp_snr_matched_filters', snr_matched_filters)
np.savetxt('periodogram_pop/temp_extension_factors', extension_factors)

outdir = "periodogram_pop"
Path(outdir).mkdir(parents=True, exist_ok=True)

plt.xlabel('f [Hz]')
plt.ylabel('p(f)')
plt.xlim(0.2, 1.2)
plt.legend()
# plt.savefig(f'{outdir}/whittle_{modes[mode]}_frequency_histograms.png')
plt.show()

plt.plot(extension_factors, snr_optimals, label='Optimal SNR')
plt.plot(extension_factors, snr_matched_filters, label='Matched Filter SNR')
plt.xlabel(r'Extension factor $x$')
plt.ylabel(r'SNR')
plt.legend()
plt.savefig(f'periodogram_pop/snr_vs_extension_{modes[mode]}.pdf')
plt.show()

assert False

ln_bfs_zero_padded.reverse()
ln_bfs.reverse()
durations_reduced.reverse()
log_frequency_spreads_zero_padded.reverse()
log_frequency_spreads.reverse()

prop_ln_bfs = ln_bfs_zero_padded[0]/durations_reduced[0] * np.array(durations_reduced)
plt.plot(durations_reduced, ln_bfs, color='red', label="white noise padded")
plt.plot(durations_reduced, ln_bfs_zero_padded, color='orange', label='zero padded')
plt.plot(durations_reduced, prop_ln_bfs, color='brown', label='proportional')
plt.xlabel("segment length [s]")
plt.ylabel("ln BF")
plt.axvline(durations[0], label="burst length", color='blue', linestyle='-.')
plt.legend()
plt.savefig(f'{outdir}/{injection_id}_duration_ln_bf.png')
plt.show()


plt.plot(durations_reduced, log_frequency_spreads, color='red', label='white noise padded')
plt.plot(durations_reduced, log_frequency_spreads_zero_padded, color='orange', label='zero padded')
plt.xlabel("segment length [s]")
plt.ylabel("Standard deviation of $\ln f$")
plt.axvline(burst_length, label="burst_length", color='blue', linestyle='-.')
plt.legend()
plt.savefig(f'{outdir}/{injection_id}_duration_ln_f_spread.png')
plt.show()
