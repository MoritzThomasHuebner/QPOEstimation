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
import warnings


# matplotlib.use('Qt5Agg')

modes = ['zeros', 'white_noise']
mode = 1
injection_id = str(mode + 6).zfill(2)
outdir = "periodogram_pop"
Path(outdir).mkdir(parents=True, exist_ok=True)
normalisation = False


end_times = np.append(np.arange(20, 500, 10), np.arange(500, 5000, 100))

start_times = -end_times
durations = 2 * end_times


outdir_qpo_periodogram = f'injection/general_qpo_injection/general_qpo_recovery/whittle/results/'
outdir_red_noise_periodogram = f'injection/general_qpo_injection/red_noise_recovery/whittle/results/'


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

# ln_bfs = []
# log_frequency_spreads = []
# durations_reduced = []
# snrs_optimal = []
# snrs_max_like = []
# snrs_max_like_quantiles = []
# extension_factors = []
# delta_bics = []
# chi_squares = []
# chi_squares_qpo = []
# chi_squares_red_noise = []
# chi_squares_high_freqs = []
# chi_squares_weighted = []

# for start_time, end_time, duration in zip(start_times, end_times, durations):
#     # try:
#     extension_factor = duration/durations[0]
#     duration_white_noise = duration - durations[0]
#     series = bilby.core.series.CoupledTimeAndFrequencySeries(duration=duration, sampling_frequency=sampling_frequency, start_time=start_time)
#     idxs = QPOEstimation.utils.get_indices_by_time(times=times, minimum_time=start_time, maximum_time=end_time)
#     y_selected = y[idxs]
#     if normalisation:
#         y_selected = (y_selected - np.mean(y_selected)) / np.mean(y_selected)
#
#     label_red_noise = f'{injection_id}_{float(start_time)}_{float(end_time)}'
#     label_qpo = f'{injection_id}_{float(start_time)}_{float(end_time)}'
#     try:
#         res_noise = bilby.result.read_in_result(outdir=outdir_red_noise_periodogram, label=label_red_noise)
#         res_qpo = bilby.result.read_in_result(outdir=outdir_qpo_periodogram, label=label_qpo)
#         ln_bfs.append(res_qpo.log_evidence - res_noise.log_evidence)
#         log_frequency_spreads.append(np.std(res_qpo.posterior['log_frequency']))
#         plt.hist(np.exp(res_qpo.posterior['log_frequency']), histtype='step', label=f"{duration}s",
#                  density=True, bins='fd')
#         durations_reduced.append(duration)
#         extension_factors.append(extension_factor)
#     except Exception as e:
#         print(e)
#         continue
#     if duration_white_noise != 0:
#         fd_data_white_noise, freqs_white_noise = psd_white_noise.get_noise_realisation(
#             sampling_frequency=sampling_frequency, duration=duration_white_noise)
#         td_data_white_noise = bilby.core.utils.infft(frequency_domain_strain=fd_data_white_noise,
#                                                      sampling_frequency=sampling_frequency)
#     else:
#         td_data_white_noise = np.array([])
#
#     ### Optimal SNR calculation
#     if end_time == 20 and modes[mode] == "zeros":
#         window = 'boxcar'
#     elif end_time == 20 and modes[mode] == "white_noise":
#         window = ("tukey", 0.05)
#     else:
#         window = "hann"
#     freqs_combined_periodogram, powers_combined_periodogram = \
#         periodogram(y_selected, fs=sampling_frequency, window=window)
#     if modes[mode] == 'zeros':
#         psd_array_noise_diluted = QPOEstimation.model.psd.red_noise(
#             frequencies=frequencies, alpha=alpha,
#             beta=beta) / extension_factor + white_noise / extension_factor
#     else:
#         psd_array_noise_diluted = QPOEstimation.model.psd.red_noise(
#             frequencies=frequencies, alpha=alpha,
#             beta=beta) / extension_factor + white_noise
#     psd_array_qpo_diluted = psd_array_qpo / extension_factor
#     psd_array_signal_diluted = psd_array_noise_diluted + psd_array_qpo_diluted
#     psd_noise_diluted = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
#         frequency_array=frequencies, psd_array=psd_array_noise_diluted)
#     psd_qpo_diluted = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
#         frequency_array=frequencies, psd_array=psd_array_qpo_diluted)
#     psd_signal_diluted = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
#         frequency_array=frequencies, psd_array=psd_array_signal_diluted)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         snr_optimal = np.sqrt(np.sum(
#             np.nan_to_num((psd_qpo_diluted.power_spectral_density_interpolated(freqs_combined_periodogram) /
#                            psd_noise_diluted.power_spectral_density_interpolated(freqs_combined_periodogram)) ** 2, nan=0)))
#     snrs_optimal.append(snr_optimal)
#
#     ### Inferred SNR calculation
#     bic_qpo = 6 * np.log(len(y_selected)) - 2 * res_qpo.posterior.iloc[-1]['log_likelihood']
#     bic_noise = 3 * np.log(len(y_selected)) - 2 * res_noise.posterior.iloc[-1]['log_likelihood']
#     delta_bics.append(bic_qpo - bic_noise)
#
#
#     snrs = []
#     for i in range(100):
#         params = res_qpo.posterior.iloc[np.random.randint(0, len(res_qpo.posterior))]
#         alpha_max_like = params['alpha']
#         beta_max_like = np.exp(params['log_beta'])
#         white_noise_max_like = np.exp(params['log_sigma'])
#         amplitude_max_like = np.exp(params['log_amplitude'])
#         width_max_like = np.exp(params['log_width'])
#         central_frequency_max_like = np.exp(params['log_frequency'])
#
#         psd_array_noise_max_like = QPOEstimation.model.psd.red_noise(
#             frequencies=frequencies, alpha=alpha_max_like,
#             beta=beta_max_like) + white_noise_max_like
#         psd_array_qpo_max_like = QPOEstimation.model.psd.lorentzian(
#             frequencies=frequencies, amplitude=amplitude_max_like, width=width_max_like,
#             central_frequency=central_frequency_max_like)
#         psd_array_signal_max_like = psd_array_noise_max_like + psd_array_qpo_max_like
#         psd_noise_max_like = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
#             frequency_array=frequencies, psd_array=psd_array_noise_max_like)
#         psd_qpo_max_like = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
#             frequency_array=frequencies, psd_array=psd_array_qpo_max_like)
#         psd_signal_max_like = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
#             frequency_array=frequencies, psd_array=psd_array_signal_max_like)
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             snr = np.sqrt(np.sum(
#                 np.nan_to_num((psd_qpo_max_like.power_spectral_density_interpolated(freqs_combined_periodogram) /
#                                psd_noise_max_like.power_spectral_density_interpolated(freqs_combined_periodogram)) ** 2, nan=0)))
#         snrs.append(snr)
#
#     snrs_max_like_quantiles.append(np.quantile(snrs, q=[0.05, 0.95]))
#
#
#     alpha_max_like = res_qpo.posterior.iloc[-1]['alpha']
#     beta_max_like = np.exp(res_qpo.posterior.iloc[-1]['log_beta'])
#     white_noise_max_like = np.exp(res_qpo.posterior.iloc[-1]['log_sigma'])
#     amplitude_max_like = np.exp(res_qpo.posterior.iloc[-1]['log_amplitude'])
#     width_max_like = np.exp(res_qpo.posterior.iloc[-1]['log_width'])
#     central_frequency_max_like = np.exp(res_qpo.posterior.iloc[-1]['log_frequency'])
#
#     psd_array_noise_max_like = QPOEstimation.model.psd.red_noise(
#         frequencies=frequencies, alpha=alpha_max_like,
#         beta=beta_max_like) + white_noise_max_like
#     psd_array_qpo_max_like = QPOEstimation.model.psd.lorentzian(
#         frequencies=frequencies, amplitude=amplitude_max_like, width=width_max_like,
#         central_frequency=central_frequency_max_like)
#     psd_array_signal_max_like = psd_array_noise_max_like + psd_array_qpo_max_like
#     psd_noise_max_like = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
#         frequency_array=frequencies, psd_array=psd_array_noise_max_like)
#     psd_qpo_max_like = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
#         frequency_array=frequencies, psd_array=psd_array_qpo_max_like)
#     psd_signal_max_like = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
#         frequency_array=frequencies, psd_array=psd_array_signal_max_like)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         snr_max_like = np.sqrt(np.sum(
#             np.nan_to_num((psd_qpo_max_like.power_spectral_density_interpolated(freqs_combined_periodogram) /
#                            psd_noise_max_like.power_spectral_density_interpolated(freqs_combined_periodogram)) ** 2, nan=0)))
#     snrs_max_like.append(snr_max_like)
#     # Remove this line again
#     snrs_max_like_quantiles.append([snrs_max_like[-1] - 0.1, snrs_max_like[-1] + 0.1])
#
#     # plt.loglog()
#     # plt.step(freqs_combined_periodogram[1:], powers_combined_periodogram[1:])
#     # plt.loglog(psd_signal_max_like.frequency_array, psd_signal_max_like.psd_array)
#     # plt.show()
#     ### Chi-squared tests
#     idxs = QPOEstimation.utils.get_indices_by_time(freqs_combined_periodogram,
#                                                    minimum_time=1/sampling_frequency-0.0001,
#                                                    maximum_time=sampling_frequency/2)
#     dofs = len(idxs) - 6
#
#
#
#     chi_squares.append(QPOEstimation.model.psd.periodogram_chi_square_test(
#         frequencies=freqs_combined_periodogram[idxs], powers=powers_combined_periodogram[idxs],
#         psd=psd_signal_max_like, degrees_of_freedom=dofs))
#
#     idxs = QPOEstimation.utils.get_indices_by_time(freqs_combined_periodogram,
#                                                    minimum_time=central_frequency_max_like-2*width_max_like,
#                                                    maximum_time=central_frequency_max_like+2*width_max_like)
#
#     dofs = len(idxs) - 6
#     chi_squares_qpo.append(QPOEstimation.model.psd.periodogram_chi_square_test(
#         frequencies=freqs_combined_periodogram[idxs], powers=powers_combined_periodogram[idxs],
#         psd=psd_signal_max_like, degrees_of_freedom=dofs))
#
#     frequency_break_max_like = (beta_max_like/white_noise_max_like/extension_factor)**(1/alpha_max_like)
#     idxs = QPOEstimation.utils.get_indices_by_time(freqs_combined_periodogram,
#                                                    minimum_time=1/sampling_frequency,
#                                                    maximum_time=frequency_break_max_like)
#     dofs = len(idxs) - 6
#     chi_squares_red_noise.append(QPOEstimation.model.psd.periodogram_chi_square_test(
#         frequencies=freqs_combined_periodogram[idxs], powers=powers_combined_periodogram[idxs],
#         psd=psd_signal_max_like, degrees_of_freedom=dofs))
#     print(extension_factor)
#     # print(snrs_max_like_quantiles[-1][0])
#     # print(snrs_max_like_quantiles[-1][1])
#     # print(frequency_break_max_like)
#     # print(len(idxs))
#     # print(chi_squares_red_noise[-1])
#
#     idxs = QPOEstimation.utils.get_indices_by_time(freqs_combined_periodogram,
#                                                    minimum_time=central_frequency_max_like + width_max_like * 2,
#                                                    maximum_time=np.max(freqs_combined_periodogram))
#     dofs = len(idxs) - 6
#
#
#     chi_squares_high_freqs.append(QPOEstimation.model.psd.periodogram_chi_square_test(
#         frequencies=freqs_combined_periodogram[idxs], powers=powers_combined_periodogram[idxs],
#         psd=psd_signal_max_like, degrees_of_freedom=dofs))
#
#     print()
#     # except Exception as e:
#     #     print(e)
#
#


# ln_bfs = np.array(ln_bfs)
# log_frequency_spreads = np.array(log_frequency_spreads)
# durations_reduced = np.array(durations_reduced)
# snrs_optimal = np.array(snrs_optimal)
# snrs_max_like = np.array(snrs_max_like)
# snrs_max_like_quantiles = np.array(snrs_max_like_quantiles)
# extension_factors = np.array(extension_factors)
# delta_bics = np.array(delta_bics)
#
#
# np.savetxt(f"{outdir}/{injection_id}_ln_bfs.txt", ln_bfs)
# np.savetxt(f"{outdir}/{injection_id}_log_frequency_spreads.txt", log_frequency_spreads)
# np.savetxt(f"{outdir}/{injection_id}_durations_reduced.txt", durations_reduced)
# np.savetxt(f"{outdir}/{injection_id}_snrs_optimal.txt", snrs_optimal)
# np.savetxt(f"{outdir}/{injection_id}_snrs_max_like.txt", snrs_max_like)
# np.savetxt(f"{outdir}/{injection_id}_snrs_max_like_sigma.txt", snrs_max_like_quantiles)
# np.savetxt(f"{outdir}/{injection_id}_extension_factors.txt", extension_factors)
# np.savetxt(f"{outdir}/{injection_id}_delta_bics.txt", delta_bics)
# np.savetxt(f"{outdir}/{injection_id}_chi_squares.txt", chi_squares)
# np.savetxt(f"{outdir}/{injection_id}_chi_squares_qpo.txt", chi_squares_qpo)
# np.savetxt(f"{outdir}/{injection_id}_chi_squares_red_noise.txt", chi_squares_red_noise)
# np.savetxt(f"{outdir}/{injection_id}_chi_squares_high_freqs.txt", chi_squares_high_freqs)

ln_bfs = np.loadtxt(f"{outdir}/{injection_id}_ln_bfs.txt")
log_frequency_spreads = np.loadtxt(f"{outdir}/{injection_id}_log_frequency_spreads.txt")
durations_reduced = np.loadtxt(f"{outdir}/{injection_id}_durations_reduced.txt")
snrs_optimal = np.loadtxt(f"{outdir}/{injection_id}_snrs_optimal.txt")
snrs_max_like = np.loadtxt(f"{outdir}/{injection_id}_snrs_max_like.txt")
snrs_max_like_sigma = np.loadtxt(f"{outdir}/{injection_id}_snrs_max_like_sigma.txt")
extension_factors = np.loadtxt(f"{outdir}/{injection_id}_extension_factors.txt")
delta_bics = np.loadtxt(f"{outdir}/{injection_id}_delta_bics.txt")
chi_squares = np.loadtxt(f"{outdir}/{injection_id}_chi_squares.txt")
chi_squares_qpo = np.loadtxt(f"{outdir}/{injection_id}_chi_squares_qpo.txt")
chi_squares_red_noise = np.loadtxt(f"{outdir}/{injection_id}_chi_squares_red_noise.txt")
chi_squares_high_freqs = np.loadtxt(f"{outdir}/{injection_id}_chi_squares_high_freqs.txt")


plt.xlabel('f [Hz]')
plt.ylabel('p(f)')
# plt.xlim(0.2, 1.2)
plt.legend()
plt.savefig(f'{outdir}/{injection_id}_whittle_frequency_histograms.png')
plt.show()
plt.clf()

plt.plot(extension_factors, chi_squares, label='Entire spectrum')
plt.plot(extension_factors, chi_squares_qpo, label='$f_0 \pm 2\sigma$')
plt.plot(extension_factors, chi_squares_red_noise, label='$f \leq f_{\mathrm{break}, x}$')
plt.plot(extension_factors, chi_squares_high_freqs, label="$f \geq f_0 + 2\sigma$")
plt.xlabel(r'Extension factor $x$')
plt.ylabel(r"$\chi^2$")
plt.legend()
plt.savefig(f'{outdir}/{injection_id}_chi_squares_vs_extension.pdf')
plt.show()
plt.clf()

plt.plot(extension_factors, snrs_optimal, label='Optimal SNR')
plt.plot(extension_factors, snrs_max_like, label='Maximum likelihood SNR')
plt.fill_between(extension_factors, snrs_max_like_quantiles[:, 0], snrs_max_like_quantiles[:, 1], color='#ff7f0e', alpha=0.3)
if modes[mode] == 'white_noise':
    plt.axvline(x_break, color='black', linestyle='-.', label='$x_{\mathrm{break}}$')
plt.xlabel(r'Extension factor $x$')
plt.ylabel(r'SNR')
plt.legend()
plt.savefig(f'{outdir}/{injection_id}_snr_vs_extension.pdf')
plt.show()
plt.clf()



prop_ln_bfs = ln_bfs[0] * np.array(extension_factors)
plt.plot(extension_factors, ln_bfs, color='red')
plt.plot(extension_factors, prop_ln_bfs, color='blue', label='proportional')
if modes[mode] == 'white_noise':
    plt.axvline(x_break, color='black', linestyle='-.', label='$x_{\mathrm{break}}$')
plt.xlabel("extension factor")
plt.ylabel("ln BF")
plt.legend()
plt.savefig(f'{outdir}/{injection_id}_ln_bf_vs_extension.png')
plt.show()
plt.clf()


predicted_extended_delta_bics = delta_bics[0] * extension_factors + (3 - 6) * np.log(extension_factors)
plt.plot(extension_factors, delta_bics, color='red', label='Inferred $\Delta BIC$')
plt.plot(extension_factors, predicted_extended_delta_bics, color='blue', label='Predicted $\Delta BIC$')
if modes[mode] == 'white_noise':
    plt.axvline(x_break, color='black', linestyle='-.', label='$x_{\mathrm{break}}$')
plt.xlabel("extension factor")
plt.ylabel("$\Delta BIC$")
plt.legend()
plt.savefig(f'{outdir}/{injection_id}_delta_bic_vs_extension_factor.png')
plt.show()
plt.clf()


plt.plot(extension_factors, log_frequency_spreads, color='red')
plt.xlabel("extension factor")
plt.ylabel("Standard deviation of $\ln f$")
if modes[mode] == 'white_noise':
    plt.axvline(x_break, color='black', linestyle='-.', label='$x_{\mathrm{break}}$')
plt.legend()
plt.savefig(f'{outdir}/{injection_id}_ln_f_spread_vs_extension_factor.png')
plt.show()
plt.clf()
