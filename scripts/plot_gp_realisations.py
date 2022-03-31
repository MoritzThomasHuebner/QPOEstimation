import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import celerite
from scipy.signal import periodogram

import QPOEstimation

plt.style.use("paper.mplstyle")

xs = np.linspace(-1, 1, 5000)
dx = xs[1] - xs[0]
fs = 1/dx
psd_freqs = np.linspace(1, 32, 5000)

yerr = np.zeros(len(xs))

kernel_red_noise = QPOEstimation.likelihood.get_kernel(kernel_type="red_noise", jitter_term=False)
kernel_qpo = QPOEstimation.likelihood.get_kernel(kernel_type="pure_qpo", jitter_term=False)
kernel_qpo_plus_red_noise = QPOEstimation.likelihood.get_kernel(kernel_type="qpo_plus_red_noise", jitter_term=False)
kernel_qpo_plus_red_noise.set_parameter(name="terms[0]:log_a", value=0.0)
kernel_qpo.set_parameter(name="log_a", value=0.0)
kernel_qpo.set_parameter(name="log_c", value=0.0)

figure, axes = plt.subplots(ncols=2, nrows=3, sharex='col', sharey='none', figsize=(12, 8))
axes = axes.ravel()


gp = celerite.GP(kernel=kernel_red_noise, mean=0)
gp.compute(t=xs, yerr=yerr)
yss = gp.sample(3)

for ys in yss:
    axes[0].plot(xs, ys)
# axes[0].set_xlabel("time [s]")
axes[0].set_ylabel("y")

psd = gp.kernel.get_psd(psd_freqs * 2 * np.pi)
axes[1].plot(psd_freqs, psd)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_ylabel("PSD")


kernel_qpo.set_parameter(name="log_f", value=np.log(8))
kernel_qpo.set_parameter(name="log_a", value=0.0)
kernel_qpo.set_parameter(name="log_c", value=0.0)
gp = celerite.GP(kernel=kernel_qpo, mean=0)
gp.compute(t=xs, yerr=yerr)
yss = gp.sample(3)

for ys in yss:
    axes[2].plot(xs, ys)
# axes[2].set_xlabel("time [s]")
axes[2].set_ylabel("y")

psd = gp.kernel.get_psd(psd_freqs * 2 * np.pi)
axes[3].plot(psd_freqs, psd)
axes[3].set_xscale("log")
axes[3].set_yscale("log")
axes[3].set_ylabel("PSD")


kernel_qpo_plus_red_noise.set_parameter(name="terms[0]:log_f", value=np.log(8))
kernel_qpo_plus_red_noise.set_parameter(name="terms[0]:log_a", value=0.0)
kernel_qpo_plus_red_noise.set_parameter(name="terms[0]:log_c", value=0.0)
kernel_qpo_plus_red_noise.set_parameter(name="terms[1]:log_a", value=2.0)
kernel_qpo_plus_red_noise.set_parameter(name="terms[1]:log_c", value=0.0)
gp = celerite.GP(kernel=kernel_qpo_plus_red_noise, mean=0)
gp.compute(t=xs, yerr=yerr)
yss = gp.sample(3)

for ys in yss:
    axes[4].plot(xs, ys)
axes[4].set_xlabel("time [s]")
axes[4].set_ylabel("y")

psd = gp.kernel.get_psd(psd_freqs * 2 * np.pi)
axes[5].plot(psd_freqs, psd)
axes[5].set_xscale("log")
axes[5].set_yscale("log")
axes[5].set_xlabel("frequency [Hz]")
axes[5].set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
axes[5].set_ylabel("PSD")
plt.subplots_adjust(wspace=0.27, hspace=0.04)
plt.savefig("results/gp_realisations.pdf", bbox_inches="tight", dpi=150)
plt.show()