import matplotlib
import matplotlib.pyplot as plt

import QPOEstimation
from QPOEstimation.utils import *
from stingray.powerspectrum import Powerspectrum
from stingray.lightcurve import Lightcurve

matplotlib.use('Qt5Agg')
plt.style.use('paper.mplstyle')

data = np.loadtxt(f'injection_files_pop/general_qpo/whittle/05_data.txt')

times = data[:, 0]
y = data[:, 1]
sampling_frequency = int(round(1/(times[1] - times[0])))



plt.figure(dpi=150)
plt.plot(times, y)
plt.xlim(-40, 40)
plt.xlabel("time [s]")
plt.ylabel("amplitude [AU]")
plt.tight_layout()
plt.savefig("paper_figures/data_generation_example_white_noise_padding.pdf")
plt.show()


data = np.loadtxt(f'injection_files_pop/general_qpo/whittle/04_data.txt')

times = data[:, 0]
y = data[:, 1]

plt.figure(figsize=(12, 6))
for duration, label in zip([40, 80, 200, 520], ['stationary data', '1x zero padded', '4x zero padded', '12x zero padded']):
    indices = QPOEstimation.utils.get_indices_by_time(times=times, minimum_time=-duration/2, maximum_time=duration/2)
    cropped_y = y[indices]
    cropped_times = times[indices]
    lc = Lightcurve(time=cropped_times, counts=cropped_y, err=np.ones(len(cropped_times)))
    ps = Powerspectrum(lc=lc, norm='leahy')
    freqs = ps.freq
    powers = ps.power
    # freqs, powers = periodogram(cropped_y, fs=sampling_frequency, window='hann')
    plt.loglog()
    df = freqs[1] - freqs[0]
    plt.step(freqs[1:], powers[1:], label=label, where='mid')
    plt.xlim(0.7, 1.5)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Power [Arb. units]')
plt.axvline(1, label='QPO frequency', color='black', linestyle='--')
plt.legend(loc='lower left', ncol=2)
plt.xticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
plt.ylim(5e-2, 2e5)
plt.tight_layout()
plt.savefig('paper_figures/example_zero_padding_effects.pdf')
plt.show()

