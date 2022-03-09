import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset
from scipy.signal import spectrogram

import QPOEstimation
from QPOEstimation.get_data import get_data
plt.style.use("paper.mplstyle")
# matplotlib.use("Qt5Agg")

start_time = 73020
end_time = 75780

times, y, _, outdir, label = get_data(
    data_source="solar_flare", run_mode="select_time", start_time=start_time,
    end_time=end_time,
    likelihood_model="whittle", solar_flare_folder="goes", solar_flare_id="go1520130512")
y = (y - np.mean(y)) / np.mean(y)


inset_indices = QPOEstimation.utils.get_indices_by_time(minimum_time=74700, maximum_time=74900, times=times)
inset_times = times[inset_indices]
inset_y = y[inset_indices]
times -= start_time
times /= 60
inset_times -= start_time
inset_times /= 60


fig, ax1 = plt.subplots()
ax1.plot(times, y)  # , label="Normalised flux [AU]")#"x", c="b", mew=2, alpha=0.8, label="Experiment")
ax1.set_xlabel(r"Minutes after 20:17 UTC")
ax1.set_ylabel(r"Normalised flux [arb. units]")
ax1.set_xlim(0, times[-1])
ax1.set_title(r"GOES 1-8 $\mathrm{\AA}$")
# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
left = inset_times[0]
bottom = np.min(inset_y)
width = inset_times[-1] - inset_times[0]
height = np.max(inset_y) - np.min(inset_y)
ax2 = plt.axes([left, bottom, width, height])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.4, 0.1, 0.5, 0.5])
ax2.set_axes_locator(ip)
ax2.set_xticks([])
ax2.set_yticks([])
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
ax2.plot(inset_times, inset_y)#, "x", c="b", mew=2, alpha=0.8, label="Experiment")
mark_inset(ax1, ax2, loc1=1, loc2=2, fc="none", ec="0.5", zorder=4)


# ax1.set_ylim(0, 26)
# ax2.set_yticks(np.arange(0, 2, 0.4))
# ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor="w")
# ax2.tick_params(axis="x", which="major", pad=8)
plt.tight_layout()
plt.savefig("results/paper_figures/goes_time_series.pdf")
plt.clf()
# plt.show()

start_time = 73020 - 11040
end_time = 75800 + 11040
times, y, _, outdir, label = get_data(
    data_source="solar_flare", run_mode="select_time", start_time=start_time,
    end_time=end_time,
    likelihood_model="whittle", solar_flare_folder="goes", solar_flare_id="go1520130512")
y = (y - np.mean(y)) / np.mean(y)
fs = 1/(times[1] - times[0])

for segment_length_spectrogram in [200, 1000, 2760]:
    nperseg = int(segment_length_spectrogram * fs)
    f, t, s_xx = spectrogram(x=y, fs=fs, nperseg=nperseg, noverlap=int(0.9*nperseg))
    # t -= nperseg
    t /= 60
    plt.pcolormesh(t, f, np.log10(s_xx), shading="auto")
    plt.colorbar()
    plt.ylabel(r"Frequency [Hz]")
    plt.xlabel(r"Minutes after 17:36 UTC")
    # plt.xticks([])
    plt.plot((74800 - start_time)/60, 1 / 12.6, color="red", marker="+", markersize=6, markeredgewidth=1.5)
    plt.savefig(f"results/paper_figures/spectrogram_solar_flare_long_overlapping_{segment_length_spectrogram}.pdf")
    # plt.show()
    plt.clf()
