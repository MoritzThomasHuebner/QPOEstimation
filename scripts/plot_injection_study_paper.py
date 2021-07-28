import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset, inset_axes, zoomed_inset_axes
from stingray.powerspectrum import Powerspectrum
from stingray.lightcurve import Lightcurve

import QPOEstimation
# from scipy.signal import periodogram
from scipy.signal.windows import hann

plt.style.use('paper.mplstyle')
# matplotlib.use('Qt5Agg')


data = np.loadtxt('injection_files_pop/general_qpo/whittle/01_data.txt')
times = data[:, 0]
y = data[:, 1]

indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-30, maximum_time=30, times=times)
times = times[indices]
y = y[indices]


fig, ax1 = plt.subplots()
ax1.plot(times, y)
ax1.set_xlabel(r'times [s]')
ax1.set_ylabel(r'Flux [Arb. units]')
plt.tight_layout()
plt.savefig('paper_figures/01_injection_time_series.pdf')
plt.show()
plt.clf()


data = np.loadtxt('injection_files_pop/general_qpo/whittle/01_data.txt')
times = data[:, 0]
y = data[:, 1]

indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-20, maximum_time=20, times=times)
times = times[indices]
y = y[indices]


lc = Lightcurve(time=times, counts=y * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm='leahy')
freqs = ps.freq
powers = ps.power

plt.loglog()
plt.step(freqs[1:], powers[1:], where='mid')
plt.xlabel('frequency [Hz]')
plt.ylabel('Power [arb. units]')
plt.axvline(1, label='QPO frequency', ls='--', color='black')
plt.legend()
plt.savefig('paper_figures/01_injection_periodogram.pdf')
plt.show()
plt.clf()

### Flare injection 02

data = np.loadtxt('injection_files_pop/general_qpo/whittle/02_data.txt')
times = data[:, 0]
y = data[:, 1]

inset_indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-10, maximum_time=10, times=times)
inset_times = times[inset_indices]
inset_y = y[inset_indices]


fig, ax1 = plt.subplots()
ax1.plot(times, y)
ax1.set_xlabel(r'times [s]')
ax1.set_ylabel(r'Flux [arb. units]')

left = inset_times[0]
bottom = np.min(inset_y)
width = inset_times[-1] - inset_times[0]
height = np.max(inset_y) - np.min(inset_y)
ax2 = plt.axes([left, bottom, width, height])

ip = InsetPosition(ax1, [0.55, 0.55, 0.4, 0.4])
ax2.set_axes_locator(ip)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.plot(inset_times, inset_y)
mark_inset(ax1, ax2, loc1=2, loc2=4, fc='none', ec='0.5', zorder=4)

plt.tight_layout()
plt.savefig('paper_figures/02_injection_time_series.pdf')
plt.show()
plt.clf()


lc = Lightcurve(time=inset_times, counts=inset_y * hann(len(inset_y)), err=np.ones(len(inset_times)))
ps = Powerspectrum(lc=lc, norm='leahy')
freqs = ps.freq
powers = ps.power

plt.loglog()
plt.step(freqs[1:], powers[1:], where='mid')
plt.xlabel('frequency [Hz]')
plt.ylabel('Power [arb. units]')
plt.axvline(1, label='QPO frequency', ls='--', color='black')
plt.legend()
plt.tight_layout()
plt.savefig('paper_figures/02_injection_periodogram.pdf')
plt.show()
plt.clf()

lc = Lightcurve(time=times, counts=y * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm='leahy')
freqs = ps.freq
powers = ps.power

plt.loglog()
plt.step(freqs[1:], powers[1:], where='mid')
plt.xlabel('frequency [Hz]')
plt.ylabel('Power [arb. units]')
plt.axvline(1, label='QPO frequency', ls='--', color='black')
plt.legend()
plt.tight_layout()
plt.savefig('paper_figures/02_injection_periodogram_total.pdf')
plt.show()
plt.clf()


data = np.loadtxt('injection_files_pop/general_qpo/whittle/09_data.txt')
times = data[:, 0]
y = data[:, 1]

indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-20, maximum_time=20, times=times)
times = times[indices]
y = y[indices]


fig, ax1 = plt.subplots()
ax1.plot(times, y)
ax1.set_xlabel(r'times [s]')
ax1.set_ylabel(r'Flux [arb. units]')
plt.tight_layout()
plt.savefig('paper_figures/09_injection_time_series.pdf')
plt.show()
plt.clf()


data = np.loadtxt('injection_files_pop/general_qpo/whittle/09_data.txt')
times = data[:, 0]
y = data[:, 1]

lc = Lightcurve(time=times, counts=(y + np.mean(y) + 1) * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm='leahy')
freqs = ps.freq
powers = ps.power


plt.loglog()
plt.step(freqs[1:], powers[1:], where='mid')
plt.xlabel('frequency [Hz]')
plt.ylabel('Power [arb. units]')
plt.axvline(1, label='QPO frequency', ls='--', color='black')
plt.legend()
plt.savefig('paper_figures/09_injection_periodogram.pdf')
plt.show()
plt.clf()

data = np.loadtxt('injection_files_pop/general_qpo/whittle/10_data.txt')
times = data[:, 0]
y = data[:, 1]

indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-20, maximum_time=20, times=times)
times = times[indices]
y = y[indices]


fig, ax1 = plt.subplots()
ax1.plot(times, y)
ax1.set_xlabel(r'times [s]')
ax1.set_ylabel(r'Flux [arb. units]')
plt.tight_layout()
plt.savefig('paper_figures/10_injection_time_series.pdf')
plt.show()
plt.clf()

data = np.loadtxt('injection_files_pop/general_qpo/whittle/10_data.txt')
times = data[:, 0]
y = data[:, 1]

lc = Lightcurve(time=times, counts=(y + np.mean(y) + 10000) * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm='leahy')
freqs = ps.freq
powers = ps.power

plt.loglog()
plt.step(freqs[1:], powers[1:], where='mid')
plt.xlabel('frequency [Hz]')
plt.ylabel('Power [arb. units]')
plt.axvline(1, label='QPO frequency', ls='--', color='black')
plt.legend()
plt.savefig('paper_figures/10_injection_periodogram_extended.pdf')
plt.show()
plt.clf()

indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-10, maximum_time=10, times=times)
times = times[indices]
y = y[indices]

lc = Lightcurve(time=times, counts=(y + np.mean(y) + 10000) * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm='leahy')
freqs = ps.freq
powers = ps.power

plt.loglog()
plt.step(freqs[1:], powers[1:], where='mid')
plt.xlabel('frequency [Hz]')
plt.ylabel('Power [arb. units]')
plt.axvline(1, label='QPO frequency', ls='--', color='black')
plt.legend()
plt.savefig('paper_figures/10_injection_periodogram.pdf')
plt.show()
plt.clf()






data = np.loadtxt('injection_files_pop/general_qpo/whittle/11_data.txt')
times = data[:, 0]
y = data[:, 1]

inset_indices = QPOEstimation.utils.get_indices_by_time(minimum_time=-10, maximum_time=10, times=times)
inset_times = times[inset_indices]
inset_y = y[inset_indices]


fig, ax1 = plt.subplots()
ax1.plot(times, y)
ax1.set_xlabel(r'times [s]')
ax1.set_ylabel(r'Flux [arb. units]')

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
mark_inset(ax1, ax2, loc1=2, loc2=4, fc='none', ec='0.5', zorder=4)


plt.tight_layout()
plt.savefig('paper_figures/11_injection_time_series.pdf')
plt.show()
plt.clf()

lc = Lightcurve(time=times, counts=y * hann(len(y)), err=np.ones(len(times)))
ps = Powerspectrum(lc=lc, norm='leahy')
freqs = ps.freq
powers = ps.power


plt.loglog()
plt.step(freqs[1:], powers[1:], where='mid')
plt.xlabel('frequency [Hz]')
plt.ylabel('Power [arb. units]')
plt.axvline(5, label='QPO frequency', ls='--', color='black')
plt.axvline(0.5, label='Low frequency cut off', ls='-.', color='green')
plt.legend()
plt.savefig('paper_figures/11_injection_periodogram.pdf')
plt.show()
plt.clf()

data = np.loadtxt('injection_files_pop/general_qpo/whittle/11_data.txt')
times = data[:, 0]
y = data[:, 1]

