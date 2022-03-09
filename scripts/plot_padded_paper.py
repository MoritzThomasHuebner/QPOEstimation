import matplotlib
import matplotlib.pyplot as plt

import QPOEstimation
from QPOEstimation.utils import *
from stingray.powerspectrum import Powerspectrum
from stingray.lightcurve import Lightcurve
from scipy.signal import periodogram

# matplotlib.use("Qt5Agg")
plt.style.use("paper.mplstyle")

data = np.loadtxt(f"injections/injection_files_pop/qpo_plus_red_noise/whittle/00_data.txt")

times = data[:, 0]
y = data[:, 1]
sampling_frequency = int(round(1/(times[1] - times[0])))


fig, ax = plt.subplots(dpi=150)
ax.plot(times, y)
ax.set_xlabel("time [s]")
ax.set_ylabel("Flux [arb. units]")
ax.set_xlim(-40, 40)

ax2 = ax.twiny()
ax2.set_xlabel("$x$")
ax2.set_xticks(ticks=(-40, -30, -20, -10, 0, 10, 20, 30, 40))
ax2.set_xticklabels(labels=(4, 3, 2, 1, 0, 1, 2, 3, 4))

fig.tight_layout()
fig.show()
fig.savefig("results/paper_figures/data_generation_example_white_noise_padding.pdf")


# data = np.loadtxt(f"injection_files_pop/qpo_plus_red_noise/whittle/00_data.txt")

times = data[:, 0]
y = data[:, 1]

# plt.figure(figsize=(13.5, 9))
# fontsize = 32
# plt.rcParams.update({"axes.labelsize": fontsize, "axes.titlesize": fontsize, "xtick.labelsize": fontsize, "ytick.labelsize": fontsize, "legend.fontsize": fontsize, "font.size": fontsize})
# plt.rc("font", size=22)          # controls default text sizes
# plt.rc("axes", titlesize=22)     # fontsize of the axes title
# plt.rc("axes", labelsize=22)    # fontsize of the x and y labels
# plt.rc("xtick", labelsize=22)    # fontsize of the tick labels
# plt.rc("ytick", labelsize=22)    # fontsize of the tick labels
# plt.rc("legend", fontsize=22)    # legend fontsize
# plt.rc("figure", titlesize=22)  # fontsize of the figure title

# plt.figure(dpi=150)
for duration, label, ls in zip([20, 40, 80, 160], ["$x=1$", "$x=2$", "$x=4$", "$x=8$"], ["solid", "dotted", "dashed", "dashdot"]):
    indices = QPOEstimation.utils.get_indices_by_time(times=times, minimum_time=-duration/2, maximum_time=duration/2)
    cropped_y = y[indices]
    cropped_times = times[indices]
    # lc = Lightcurve(time=cropped_times, counts=cropped_y, err=np.ones(len(cropped_times)))
    # ps = Powerspectrum(lc=lc, norm="leahy")
    # freqs = ps.freq
    # powers = ps.power
    freqs, powers = periodogram(cropped_y, fs=sampling_frequency, window="boxcar")
    plt.loglog()
    df = freqs[1] - freqs[0]
    plt.plot(freqs[1:], powers[1:], label=label, linestyle=ls, linewidth=2.5)  # , where="mid"
    plt.xlim(0.7, 1.3)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("Power [arb. units]")
plt.axvline(1, label="QPO frequency", color="black", linestyle="solid")
plt.legend(loc="lower left", ncol=2)
plt.xticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
plt.ylim(1e-3, 1e2)
plt.tight_layout()
plt.savefig("results/paper_figures/example_zero_padding_effects.pdf")
plt.show()
