import matplotlib.pyplot as plt
import numpy as np

from QPOEstimation.model.mean import skew_gaussian, skew_exponential, fred, fred_extended
plt.style.use("paper.mplstyle")

times = np.linspace(0, 6.5, 2000)

log_amplitude = 0
log_sigma_rise = np.log(0.2)
log_sigma_fall = np.log(0.4)
t_0 = 1

ys_0 = skew_gaussian(times=times, log_amplitude=log_amplitude, t_0=t_0,
                     log_sigma_rise=log_sigma_rise, log_sigma_fall=log_sigma_fall)
ys_1 = skew_exponential(times=times, log_amplitude=log_amplitude, t_0=t_0 + 1.5,
                        log_sigma_rise=log_sigma_rise, log_sigma_fall=log_sigma_fall)
ys_2 = fred(times=times, log_amplitude=log_amplitude, log_psi=3, t_0=t_0 + 3, delta=0)
# ys_3 = fred_extended(times=times, log_amplitude=log_amplitude, log_psi=3,
#                      t_0=t_0 + 4.5, delta=0, log_gamma=0.2, log_nu=0.2)
# ys_3 /= max(ys_3)

plt.plot(times, ys_0, label="skew-Gaussian")
plt.plot(times, ys_1, label="skew-Exponential")
plt.plot(times, ys_2, label="FRED")
# plt.plot(times, ys_3, label="FRED-x")
plt.legend(ncol=2, loc="upper center")
plt.xlabel("time [s]")
plt.ylabel("y")
plt.ylim(-0.05, 1.4)
plt.tight_layout()
plt.savefig("results/realisations_mean_model.pdf")
plt.show()
