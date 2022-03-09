from QPOEstimation.stabilisation import bar_lev
from QPOEstimation.smoothing import two_sided_exponential_smoothing

import numpy as np

import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Qt5Agg")

alpha = 0.04
sampling_frequency = 128
data = np.loadtxt(f"data/sgr1806_{sampling_frequency}Hz.dat")
print(data)
times = data[:, 0]
counts = data[:, 1]
ys = bar_lev(counts)
ys_smoothed = two_sided_exponential_smoothing(ys, alpha=alpha)
ys_residual = ys - ys_smoothed
plt.plot(times, ys)
plt.plot(times, ys_smoothed)
plt.show()

plt.plot(times, ys_residual)
plt.show()

smoothed_data = np.array([times, ys_smoothed]).T
residual_data = np.array([times, ys_residual]).T
np.savetxt(f"data/sgr1806_{sampling_frequency}Hz_exp_smoothed_alpha_{alpha}.dat", smoothed_data)

res_data = np.array([times, ys_residual]).T
np.savetxt(f"data/sgr1806_{sampling_frequency}Hz_exp_residual_alpha_{alpha}.dat", residual_data)

print(times)