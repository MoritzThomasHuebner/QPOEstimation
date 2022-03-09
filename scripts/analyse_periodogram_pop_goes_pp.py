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
from QPOEstimation.get_data import get_data


matplotlib.use("Qt5Agg")



outdir_qpo_periodogram = f"results/solar_flare_go1520130512/select_time/qpo_plus_red_noise/whittle/results/"
outdir_red_noise_periodogram = f"results/solar_flare_go1520130512/select_time/red_noise/whittle/results/"
outdir_bpl_periodogram = f"results/solar_flare_go1520130512/select_time/broken_power_law/whittle/results/"

outdirs = [outdir_red_noise_periodogram, outdir_qpo_periodogram, outdir_bpl_periodogram]
# outdirs = [outdir_qpo_periodogram]
ranges = ["73020_74700", "74700.0_74900.0", "74900_75780", "73020_75780"]

for r in ranges:
    ref_log_z = 0
    ref_log_z_err = 0
    ref_bic = 0
    times, y, _, outdir, label = get_data(
        data_source="solar_flare", run_mode="select_time", start_time=float(r.split("_")[0]), end_time=float(r.split("_")[1]),
        likelihood_model="whittle", solar_flare_folder="goes", solar_flare_id="go1520130512")
    y = (y - np.mean(y)) / np.mean(y)
    print(r)
    for outdir in outdirs:
        res = bilby.result.read_in_result(outdir=outdir, label=r)
        max_like_params = res.posterior.iloc[-1]
        if outdir == outdir_qpo_periodogram:
            k = 6
        elif outdir == outdir_red_noise_periodogram:
            k = 3
        else:
            k = 5

        if outdir == outdir_red_noise_periodogram:
            ref_log_z = res.log_evidence
            ref_log_z_err = res.log_evidence_err
            ref_bic = k * np.log(len(times)/2) - 2 * max_like_params["log_likelihood"]

        print("ln BF vs red noise")
        print(f"{res.log_evidence - ref_log_z} +/- {np.sqrt(res.log_evidence_err**2 + ref_log_z_err**2)}")
        print("Delta BIC vs red noise")
        bic = k * np.log(len(times) / 2) - 2 * max_like_params["log_likelihood"]
        print(f"{bic - ref_bic}")
        print()
        # continue
        if outdir == outdir_qpo_periodogram:
            samples = np.exp(-res.posterior["log_frequency"])
            samples = np.array(samples)

            if r == ranges[0]:
                plot_samples = samples[np.where(samples < 50)]
                # plt.hist(plot_samples, bins="fd")
                # plt.show()
                samples_low = samples[np.where(samples < 10)]
                samples_high = samples[np.where(np.logical_and(samples >15, samples < 30))]
                print(np.median(samples_low))
                print(np.quantile(samples_low, [0.16, 0.84]) - np.median(samples_low))
                print(np.median(samples_high))
                print(np.quantile(samples_high, [0.16, 0.84]) - np.median(samples_high))
            else:
                # plt.hist(samples, bins="fd")
                # plt.show()
                print(np.median(samples))
                print(np.quantile(samples, [0.16, 0.84]) - np.median(samples))

            sampling_frequency = 1 / (times[1] - times[0])
            # plt.plot(times, y)
            # plt.show()

            freqs, powers = periodogram(y, fs=sampling_frequency, window="hann")

            frequencies = np.linspace(1 / 100000, 2, 1000000)
            alpha = max_like_params["alpha"]
            beta = np.exp(max_like_params["log_beta"])
            white_noise = np.exp(max_like_params["log_sigma"])
            amplitude = np.exp(max_like_params["log_amplitude"])
            width = np.exp(max_like_params["log_width"])
            central_frequency = np.exp(max_like_params["log_frequency"])

            psd_array_noise = QPOEstimation.model.psd.red_noise(frequencies=frequencies, alpha=alpha,
                                                                beta=beta) + white_noise
            psd_array_qpo = QPOEstimation.model.psd.lorentzian(frequencies=frequencies, amplitude=amplitude,
                                                               width=width, central_frequency=central_frequency)
            psd_array_white_noise = white_noise * np.ones(len(frequencies))

            psd_noise = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
                frequency_array=frequencies, psd_array=psd_array_noise)
            psd_signal = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
                frequency_array=frequencies, psd_array=psd_array_noise + psd_array_qpo)
            psd_white_noise = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
                frequency_array=frequencies, psd_array=psd_array_white_noise)
            psd_qpo = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
                frequency_array=frequencies, psd_array=psd_array_qpo)
            plt.loglog()
            plt.step(freqs[1:], powers[1:])
            plt.plot(psd_signal.frequency_array, psd_signal.psd_array)
            plt.xlim(freqs[1], freqs[-1])
            plt.ylim(np.min(powers), powers[1])
            plt.show()

            ### Chi-squared tests
            dofs = len(freqs) - 6
            chi_square = QPOEstimation.model.psd.periodogram_chi_square_test(
                frequencies=freqs, powers=powers,
                psd=psd_signal, degrees_of_freedom=dofs)

            print(f"chi-square entire segment: {chi_square}")

            idxs = QPOEstimation.utils.get_indices_by_time(
                freqs, minimum_time=central_frequency - 2*width, maximum_time=central_frequency + 2*width)

            dofs = len(idxs) - 6
            chi_square_qpo = QPOEstimation.model.psd.periodogram_chi_square_test(
                frequencies=freqs[idxs], powers=powers[idxs],
                psd=psd_signal, degrees_of_freedom=dofs)

            print(f"chi-square QPO: {chi_square_qpo}")

            frequency_break = (beta / white_noise) ** (1 / alpha)
            idxs = QPOEstimation.utils.get_indices_by_time(freqs,
                                                           minimum_time=0,
                                                           maximum_time=frequency_break)
            dofs = len(idxs) - 6
            chi_square_red_noise = QPOEstimation.model.psd.periodogram_chi_square_test(
                frequencies=freqs[idxs], powers=powers[idxs],
                psd=psd_signal, degrees_of_freedom=dofs)
            print(f"chi-square Red noise: {chi_square_red_noise}")
        print()

assert False




