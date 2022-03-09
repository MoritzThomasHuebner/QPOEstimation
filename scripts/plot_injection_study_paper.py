import bilby.core.result
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset, inset_axes, zoomed_inset_axes
from stingray.powerspectrum import Powerspectrum
from stingray.lightcurve import Lightcurve

import QPOEstimation
# from scipy.signal import periodogram
from scipy.signal.windows import hann

plt.style.use("paper.mplstyle")
# matplotlib.use("Qt5Agg")


data = np.loadtxt("injections/injection_files_pop/qpo_plus_red_noise/whittle/01_data.txt")
times = data[:, 0]
y = data[:, 1]

indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-30, maximum_time=30, times=times)
times = times[indices]
y = y[indices]


fig, ax1 = plt.subplots()
ax1.plot(times, y)
ax1.set_xlabel(r"times [s]")
ax1.set_ylabel(r"Flux [arb. units]")
ax1.set_xlim(-30, 30)
ax2 = ax1.twiny()
ax2.set_xlabel("$x$")
ax2.set_xticks(ticks=(-30, -20, -10, 0, 10, 20, 30))
ax2.set_xticklabels(labels=(3, 2, 1, 0, 1, 2, 3))
plt.tight_layout()
plt.savefig("results/paper_figures/01_injection_time_series.pdf")
plt.show()
plt.clf()


data = np.loadtxt("injections/injection_files_pop/qpo_plus_red_noise/whittle/01_data.txt")
times = data[:, 0]
y = data[:, 1]

indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-10, maximum_time=10, times=times)
times = times[indices]
y = y[indices]


lc = Lightcurve(time=times, counts=y * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm="leahy")

from scipy.signal import periodogram
fs = 1/(times[1] - times[0])
freqs, powers = periodogram(y, fs=fs, window="hann")

# lc = Lightcurve(time=inset_times, counts=inset_y * hann(len(inset_y)), err=np.ones(len(inset_times)))
# ps = Powerspectrum(lc=lc, norm="leahy")

# freqs = ps.freq
# powers = ps.power

plt.loglog()
plt.step(freqs[1:], powers[1:], where="mid")
plt.xlabel("frequency [Hz]")
plt.ylabel("Power [arb. units]")
plt.axvline(1, label="QPO frequency", ls="--", color="black")
plt.legend()
plt.savefig("results/paper_figures/01_injection_periodogram.pdf")
plt.show()
plt.clf()

### Flare injection 02

data = np.loadtxt("injections/injection_files_pop/qpo_plus_red_noise/whittle/02_data.txt")
times = data[:, 0]
y = data[:, 1]

inset_indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-10, maximum_time=10, times=times)
inset_times = times[inset_indices]
inset_y = y[inset_indices]

profile_amplitude = 20000
profile_t_0 = 70
sigma_fall = 20
sigma_rise = 10

trend = QPOEstimation.model.mean.skew_exponential(
        times=inset_times + 100, log_amplitude=np.log(profile_amplitude), t_0=profile_t_0,
        log_sigma_fall=np.log(sigma_fall), log_sigma_rise=np.log(sigma_rise))

inset_y_detrended = inset_y - trend

fig, ax1 = plt.subplots()
ax1.plot(times, y)
ax1.set_xlabel(r"times [s]")
ax1.set_ylabel(r"Flux [arb. units]")
ax1.set_xlim(-100, 100)

ax4 = ax1.twiny()
ax4.set_xlabel("$x$")
ax4.set_xticks(ticks=(-100, -70, -50, -30, -10, 10, 30, 50, 70, 100))
ax4.set_xticklabels(labels=("", 7, 5, 3, 1, 1, 3, 5, 7, 10))


left = inset_times[0]
bottom = np.min(inset_y)
width = inset_times[-1] - inset_times[0]
height = np.max(inset_y) - np.min(inset_y)


ip = InsetPosition(ax1, [0.55, 0.55, 0.4, 0.4])

ax2 = inset_axes(ax1, 0.4, 0.4)
# ax2 = plt.gcf().add_axes([0.55, 0.55, 0.4, 0.4])
ax2.set_axes_locator(ip)
ax2.plot([-10, 10], [inset_y[0], inset_y[-1]])
# hide the ticks of the linked axes
ax2.set_xticks([])
ax2.set_yticks([])

mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec="0.5", zorder=4)

ax3 = plt.axes([left, bottom, width, height])
ax3.set_axes_locator(ip)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.plot(inset_times, inset_y_detrended)



plt.tight_layout()
plt.savefig("results/paper_figures/02_injection_time_series.pdf")
plt.show()
plt.clf()


lc = Lightcurve(time=inset_times, counts=inset_y * hann(len(inset_y)), err=np.ones(len(inset_times)))
ps = Powerspectrum(lc=lc, norm="leahy")
freqs = ps.freq
powers = ps.power

plt.loglog()
plt.step(freqs[1:], powers[1:], where="mid")
plt.xlabel("frequency [Hz]")
plt.ylabel("Power [arb. units]")
plt.axvline(1, label="QPO frequency", ls="--", color="black")
plt.legend()
plt.tight_layout()
plt.savefig("results/paper_figures/02_injection_periodogram.pdf")
plt.show()
plt.clf()

lc = Lightcurve(time=times, counts=y * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm="leahy")
freqs = ps.freq
powers = ps.power

plt.loglog()
plt.step(freqs[1:], powers[1:], where="mid")
plt.xlabel("frequency [Hz]")
plt.ylabel("Power [arb. units]")
plt.axvline(1, label="QPO frequency", ls="--", color="black")
plt.legend()
plt.tight_layout()
plt.savefig("results/paper_figures/02_injection_periodogram_total.pdf")
plt.show()
plt.clf()


data = np.loadtxt("injections/injection_files_pop/qpo_plus_red_noise/whittle/03_data.txt")
times = data[:, 0]
y = data[:, 1]

indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-10, maximum_time=10, times=times)
times = times[indices]
y = y[indices]


fig, ax1 = plt.subplots()
ax1.plot(times, y)
ax1.set_xlabel(r"times [s]")
ax1.set_ylabel(r"Flux [arb. units]")

plt.tight_layout()
plt.savefig("results/paper_figures/03_injection_time_series.pdf")
plt.show()
plt.clf()


data = np.loadtxt("injections/injection_files_pop/qpo_plus_red_noise/whittle/03_data.txt")
times = data[:, 0]
y = data[:, 1]

lc = Lightcurve(time=times, counts=(y + np.mean(y) + 1) * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm="leahy")
freqs = ps.freq
powers = ps.power


plt.loglog()
plt.step(freqs[1:], powers[1:], where="mid")
plt.xlabel("frequency [Hz]")
plt.ylabel("Power [arb. units]")
plt.axvline(5, label="QPO frequency", ls="--", color="black")
plt.legend()
plt.savefig("results/paper_figures/03_injection_periodogram.pdf")
plt.show()
plt.clf()

data = np.loadtxt("injections/injection_files_pop/qpo_plus_red_noise/whittle/04_data.txt")
times = data[:, 0]
y = data[:, 1]

indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-10, maximum_time=10, times=times)
times = times[indices]
y = y[indices]


fig, ax1 = plt.subplots()
ax1.plot(times, y)
ax1.set_xlabel(r"times [s]")
ax1.set_ylabel(r"Flux [arb. units]")
plt.tight_layout()
plt.savefig("results/paper_figures/04_injection_time_series.pdf")
plt.show()
plt.clf()

data = np.loadtxt("injections/injection_files_pop/qpo_plus_red_noise/whittle/04_data.txt")
times = data[:, 0]
y = data[:, 1]

lc = Lightcurve(time=times, counts=(y + np.mean(y) + 10000) * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm="leahy")
freqs = ps.freq
powers = ps.power

plt.loglog()
plt.step(freqs[1:], powers[1:], where="mid")
plt.xlabel("frequency [Hz]")
plt.ylabel("Power [arb. units]")
plt.axvline(5, label="QPO frequency", ls="--", color="black")
plt.legend()
plt.savefig("results/paper_figures/04_injection_periodogram_extended.pdf")
plt.show()
plt.clf()

indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-10, maximum_time=10, times=times)
times = times[indices]
y = y[indices]

lc = Lightcurve(time=times, counts=(y + np.mean(y) + 10000) * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm="leahy")
freqs = ps.freq
powers = ps.power

plt.loglog()
plt.step(freqs[1:], powers[1:], where="mid")
plt.xlabel("frequency [Hz]")
plt.ylabel("Power [arb. units]")
plt.axvline(5, label="QPO frequency", ls="--", color="black")
plt.legend()
plt.savefig("results/paper_figures/04_injection_periodogram.pdf")
plt.show()
plt.clf()


data = np.loadtxt("injections/injection_files_pop/qpo_plus_red_noise/whittle/05_data.txt")
times = data[:, 0]
y = data[:, 1]

inset_indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-10, maximum_time=10, times=times)
inset_times = times[inset_indices]
inset_y = y[inset_indices]


fig, ax1 = plt.subplots()
ax1.plot(times, y)
ax1.set_xlabel(r"time [s]")
ax1.set_ylabel(r"Counts")
ax1.set_xlim(-200, 200)

left = inset_times[0]
bottom = np.min(inset_y)
width = inset_times[-1] - inset_times[0]
height = np.max(inset_y) - np.min(inset_y)
ax2 = plt.axes([left, bottom, width, height])

ip = InsetPosition(ax1, [0.05, 0.2, 0.3, 0.3])
ax2.set_axes_locator(ip)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.plot(inset_times, inset_y)
mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec="0.5", zorder=4)

ax3 = ax1.twiny()
ax3.set_xlabel("$x$")
ax3.set_xticks(ticks=(-200, -150, -100, -50, -10, 10, 50, 100, 150, 200))
ax3.set_xticklabels(labels=("", 15, 10, 5, 1, 1, 5, 10, 15, 20))



plt.tight_layout()
plt.savefig("results/paper_figures/05_injection_time_series.pdf")
plt.show()
plt.clf()

lc = Lightcurve(time=times, counts=y * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm="leahy")
freqs = ps.freq
powers = ps.power


plt.loglog()
plt.step(freqs[1:], powers[1:], where="mid")
plt.xlabel("frequency [Hz]")
plt.ylabel("Power [arb. units]")
plt.axvline(5, label="QPO frequency", ls="--", color="black")
plt.axvline(0.5, label="Low frequency cut off", ls="-.", color="green")
plt.legend()
plt.savefig("results/paper_figures/05_injection_periodogram.pdf")
plt.show()
plt.clf()

data = np.loadtxt("injections/injection_files_pop/qpo_plus_red_noise/whittle/05_data.txt")
times = data[:, 0]
y = data[:, 1]
