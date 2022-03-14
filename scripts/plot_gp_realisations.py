import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import celerite

import QPOEstimation

plt.style.use("paper.mplstyle")

xs = np.linspace(-1, 1, 5000)
yerr = np.zeros(len(xs))

kernel_red_noise = QPOEstimation.likelihood.get_kernel(kernel_type="red_noise", jitter_term=False)
kernel_qpo = QPOEstimation.likelihood.get_kernel(kernel_type="pure_qpo", jitter_term=False)
kernel_qpo_plus_red_noise = QPOEstimation.likelihood.get_kernel(kernel_type="qpo_plus_red_noise", jitter_term=False)
print(kernel_red_noise.parameter_vector)
gp = celerite.GP(kernel=kernel_red_noise, mean=0)
gp.compute(t=xs, yerr=yerr)
yss = gp.sample(3)

for ys in yss:
    plt.plot(xs, ys)
plt.xlabel("Time [s]")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("results/realisations_red_noise.pdf")
plt.show()

print(kernel_qpo.parameter_names)
print(kernel_qpo.parameter_vector)
kernel_qpo.set_parameter(name="log_f", value=np.log(4))
gp = celerite.GP(kernel=kernel_qpo, mean=0)
gp.compute(t=xs, yerr=yerr)
yss = gp.sample(3)

for ys in yss:
    plt.plot(xs, ys)
plt.xlabel("Time [s]")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("results/realisations_qpo.pdf")
plt.show()



kernel_qpo_plus_red_noise.set_parameter(name="terms[0]:log_f", value=np.log(4))
kernel_qpo_plus_red_noise.set_parameter(name="terms[0]:log_a", value=0.0)
kernel_qpo_plus_red_noise.set_parameter(name="terms[1]:log_a", value=2.5)
gp = celerite.GP(kernel=kernel_qpo_plus_red_noise, mean=0)
gp.compute(t=xs, yerr=yerr)
yss = gp.sample(3)

for ys in yss:
    plt.plot(xs, ys)
plt.xlabel("Time [s]")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("results/realisations_qpo_plus_red_noise.pdf")
plt.show()
