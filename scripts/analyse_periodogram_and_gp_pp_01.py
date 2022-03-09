import json
import sys

import numpy as np
from pathlib import Path

import bilby

import QPOEstimation
from QPOEstimation.post_processing import InjectionStudyPostProcessor

import matplotlib.pyplot as plt
plt.style.use("paper.mplstyle")
# import matplotlib
# matplotlib.use("Qt5Agg")


modes = ["zeros", "white_noise"]
# mode = int(sys.argv[1])
mode = 1
injection_id = str(mode).zfill(2)
outdir = "results/periodogram_pop"
Path(outdir).mkdir(parents=True, exist_ok=True)
normalisation = False

load = True
n_snrs = 2000

end_times = np.arange(10, 210, 10)
print(end_times)

start_times = -end_times
durations = 2 * end_times


outdir_qpo_periodogram = f"injection/qpo_plus_red_noise_injection/qpo_plus_red_noise_recovery/whittle/results/"
outdir_noise_periodogram = f"injection/qpo_plus_red_noise_injection/red_noise_recovery/whittle/results/"


data = np.loadtxt(f"injection_files_pop/qpo_plus_red_noise/whittle/{injection_id}_data.txt")
times = data[:, 0]
y = data[:, 1]
sampling_frequency = int(round(1/(times[1] - times[0])))
with open(f"injection_files_pop/qpo_plus_red_noise/whittle/{injection_id}_params.json", "r") as f:
    injection_parameters = json.load(f)

frequencies = np.linspace(1/100000, 20, 1000)
alpha = injection_parameters["alpha"]
beta = injection_parameters["beta"]
white_noise = injection_parameters["sigma"]
amplitude = injection_parameters["amplitude"]
width = injection_parameters["width"]
central_frequency = injection_parameters["central_frequency"]

psd_array_noise = QPOEstimation.model.psd.red_noise(frequencies=frequencies, alpha=alpha, beta=beta) + white_noise
psd_array_white_noise = white_noise * np.ones(len(frequencies))

psd_noise = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
    frequency_array=frequencies, psd_array=psd_array_noise)
psd_white_noise = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
    frequency_array=frequencies, psd_array=psd_array_white_noise)
psd_array_qpo = QPOEstimation.model.psd.lorentzian(frequencies=frequencies, amplitude=amplitude, width=width,
                                                   central_frequency=central_frequency)
psd_qpo = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
    frequency_array=frequencies, psd_array=psd_array_qpo)


injection_psds = dict(red_noise=psd_noise, qpo=psd_qpo)
extension_mode = modes[mode]

props = InjectionStudyPostProcessor(
    start_times=start_times, end_times=end_times, durations=durations, outdir=outdir,
    label=injection_id, times=times, frequencies=frequencies, normalisation=normalisation, y=y,
    outdir_noise_periodogram=outdir_noise_periodogram, outdir_qpo_periodogram=outdir_qpo_periodogram,
    injection_parameters=injection_parameters, injection_psds=injection_psds, extension_mode=extension_mode)

if load:
    props.ln_bfs = np.loadtxt(f"{outdir}/cached_results/{injection_id}_ln_bfs.txt")
    props.log_frequency_spreads = np.loadtxt(f"{outdir}/cached_results/{injection_id}_log_frequency_spreads.txt")
    props.durations_reduced = np.loadtxt(f"{outdir}/cached_results/{injection_id}_durations_reduced.txt")
    props.snrs_optimal = np.loadtxt(f"{outdir}/cached_results/{injection_id}_snrs_optimal.txt")
    props.snrs_max_like = np.loadtxt(f"{outdir}/cached_results/{injection_id}_snrs_max_like.txt")
    props.snrs_max_like_quantiles = np.loadtxt(f"{outdir}/cached_results/{injection_id}_snrs_max_like_quantiles.txt")
    props.extension_factors = np.loadtxt(f"{outdir}/cached_results/{injection_id}_extension_factors.txt")
    props.delta_bics = np.loadtxt(f"{outdir}/cached_results/{injection_id}_delta_bics.txt")
    props.chi_squares = np.loadtxt(f"{outdir}/cached_results/{injection_id}_chi_squares.txt")
    props.chi_squares_qpo = np.loadtxt(f"{outdir}/cached_results/{injection_id}_chi_squares_qpo.txt")
    props.chi_squares_red_noise = np.loadtxt(f"{outdir}/cached_results/{injection_id}_chi_squares_red_noise.txt")
    props.chi_squares_high_freqs = np.loadtxt(f"{outdir}/cached_results/{injection_id}_chi_squares_high_freqs.txt")
else:
    props.fill(n_snrs=n_snrs)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_ln_bfs.txt", props.ln_bfs)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_log_frequency_spreads.txt", props.log_frequency_spreads)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_durations_reduced.txt", props.durations_reduced)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_snrs_optimal.txt", props.snrs_optimal)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_snrs_max_like.txt", props.snrs_max_like)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_snrs_max_like_quantiles.txt", props.snrs_max_like_quantiles)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_extension_factors.txt", props.extension_factors)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_delta_bics.txt", props.delta_bics)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_chi_squares.txt", props.chi_squares)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_chi_squares_qpo.txt", props.chi_squares_qpo)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_chi_squares_red_noise.txt", props.chi_squares_red_noise)
    np.savetxt(f"{outdir}/cached_results/{injection_id}_chi_squares_high_freqs.txt", props.chi_squares_high_freqs)

ln_bfs_gp = []
ln_bfs_gp_windowed = []
ln_zs_gp = []
ln_zs_gp_windowed = []
xs = []
min_end_time = 10
for end_time in range(10, 210, 10):
    label = f"{injection_id}_{-float(end_time)}_{float(end_time)}_1_0s"
    res_qpo = bilby.result.read_in_result(outdir="injection/qpo_plus_red_noise_injection/qpo_plus_red_noise_recovery/celerite/results/", label=label)
    res_red_noise = bilby.result.read_in_result(outdir="injection/qpo_plus_red_noise_injection/red_noise_recovery/celerite/results/", label=label)
    res_qpo_windowed = bilby.result.read_in_result(outdir="injection/qpo_plus_red_noise_injection/qpo_plus_red_noise_recovery/celerite_windowed/results/", label=label)
    res_red_noise_windowed = bilby.result.read_in_result(outdir="injection/qpo_plus_red_noise_injection/red_noise_recovery/celerite_windowed/results/", label=label)
    ln_zs_gp.append(res_qpo.log_evidence)
    ln_zs_gp_windowed.append(res_qpo_windowed.log_evidence)
    ln_bfs_gp.append(res_qpo.log_evidence - res_red_noise.log_evidence)
    ln_bfs_gp_windowed.append(res_qpo_windowed.log_evidence - res_red_noise_windowed.log_evidence)
    xs.append(end_time/min_end_time)


plt.plot(xs, props.ln_bfs, label="Periodogram")
plt.plot(xs, ln_bfs_gp, label="Stat. GP")
plt.plot(xs, ln_bfs_gp_windowed, label="Non-stat. GP")
plt.xlim(1, 20)
plt.ylim(0, 50)
plt.xlabel("$x$")
plt.ylabel("$\ln BF$")
plt.legend()
plt.tight_layout()
plt.savefig("results/thesis_figures/injection_periodogram_comparison.pdf")
# plt.show()
plt.clf()

plt.plot(xs, np.array(ln_zs_gp_windowed) - np.array(ln_zs_gp))
plt.xlim(1, 20)
plt.xlabel("$x$")
plt.ylabel("$\ln BF$")
plt.savefig("results/thesis_figures/injection_periodogram_stat_vs_non_stat.pdf")
# plt.show()
plt.clf()


# props.plot_all(show=False)
