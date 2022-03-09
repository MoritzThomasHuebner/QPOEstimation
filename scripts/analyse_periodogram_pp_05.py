from pathlib import Path

import numpy as np

from QPOEstimation.post_processing import InjectionStudyPostProcessor

import matplotlib.pyplot as plt
plt.style.use("paper.mplstyle")
# import matplotlib
# matplotlib.use("Qt5Agg")

injection_id = "05"
outdir = "results/periodogram_pop"
Path(outdir).mkdir(parents=True, exist_ok=True)
normalisation = False

load = False
n_snrs = 2000


end_times = np.arange(10, 200, 10)
start_times = -end_times
durations = 2 * end_times

outdir_qpo_periodogram = f"injection/qpo_plus_red_noise_injection/pure_qpo_recovery/whittle/results/"
outdir_noise_periodogram = f"injection/qpo_plus_red_noise_injection/white_noise_recovery/whittle/results/"

data = np.loadtxt(f"injection_files_pop/qpo_plus_red_noise/whittle/{injection_id}_data.txt")
times = data[:, 0]
y = data[:, 1]

frequencies = np.linspace(1/100000, 20, 1000)

props = InjectionStudyPostProcessor(
    start_times=start_times, end_times=end_times, durations=durations, outdir=outdir,
    label=injection_id, times=times, frequencies=frequencies,
    normalisation=normalisation, y=y, outdir_noise_periodogram=outdir_noise_periodogram,
    outdir_qpo_periodogram=outdir_qpo_periodogram)

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

props.plot_all(show=False)
