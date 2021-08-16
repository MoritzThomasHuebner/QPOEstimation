import matplotlib
import matplotlib.pyplot as plt

import QPOEstimation
from QPOEstimation.utils import *
from stingray.powerspectrum import Powerspectrum
from stingray.lightcurve import Lightcurve
from scipy.signal import periodogram

# matplotlib.use('Qt5Agg')
plt.style.use('paper.mplstyle')

data = np.loadtxt(f'injection_files_pop/general_qpo/whittle/00_data.txt')

times = data[:, 0]
y = data[:, 1]
sampling_frequency = int(round(1/(times[1] - times[0])))


fig, ax = plt.subplots(dpi=150)
ax.plot(times, y)
ax.set_xlabel('time [s]')
ax.set_ylabel('flux [arb. units]')
ax.set_xlim(-40, 40)

ax2 = ax.twiny()
ax2.set_xlabel("$x$")
ax2.set_xticks(ticks=(-40, -30, -20, -10, 0, 10, 20, 30, 40))
ax2.set_xticklabels(labels=(4, 3, 2, 1, 0, 1, 2, 3, 4))

fig.tight_layout()
fig.show()
fig.savefig("paper_figures/data_generation_example_white_noise_padding.pdf")


# data = np.loadtxt(f'injection_files_pop/general_qpo/whittle/00_data.txt')

times = data[:, 0]
y = data[:, 1]

plt.figure(figsize=(9, 6))
# plt.figure(dpi=150)
for duration, label in zip([20, 40, 80, 160], ['$x=1$', '$x=2$', '$x=4$', '$x=8$']):
    indices = QPOEstimation.utils.get_indices_by_time(times=times, minimum_time=-duration/2, maximum_time=duration/2)
    cropped_y = y[indices]
    cropped_times = times[indices]
    # lc = Lightcurve(time=cropped_times, counts=cropped_y, err=np.ones(len(cropped_times)))
    # ps = Powerspectrum(lc=lc, norm='leahy')
    # freqs = ps.freq
    # powers = ps.power
    freqs, powers = periodogram(cropped_y, fs=sampling_frequency, window='boxcar')
    plt.loglog()
    df = freqs[1] - freqs[0]
    plt.step(freqs[1:], powers[1:], label=label, where='mid')
    plt.xlim(0.7, 1.3)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('power [Arb. units]')
plt.axvline(1, label='QPO frequency', color='black', linestyle='--')
plt.legend(loc='lower left', ncol=2)
plt.xticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
plt.ylim(5e-3, 5e2)
plt.tight_layout()
plt.savefig('paper_figures/example_zero_padding_effects.pdf')
plt.show()

