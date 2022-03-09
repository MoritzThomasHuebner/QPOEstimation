import bilby
import celerite
import numpy as np
from QPOEstimation.likelihood import get_kernel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

recovery_mode = "qpo_plus_red_noise"

res = bilby.result.read_in_result(f"results/SolarFlare/120704187_two_step/results/kernel_{recovery_mode}_result.json")
max_like_parameteres = res.posterior.iloc[-1]

kernel = get_kernel(kernel_type=recovery_mode)

n_models = 1
data = np.loadtxt(f"data/SolarFlare/120704187_ctime_lc_residual_{n_models}_gaussians.txt")

times = data[:, 0]
res_y = data[:, 1]
yerr = data[:, 2]

gp = celerite.GP(kernel=kernel)
gp.compute(t=times, yerr=yerr)

for key, val in max_like_parameteres.items():
    try:
        gp.set_parameter(key, val)
    except Exception as e:
        print(e)

# gp.set_parameter("kernel:log_c", -4.7)
gp.set_parameter("kernel:log_f", np.log(1/50))

for i in range(100):
    y = gp.sample(1)[0]
    plt.plot(times-times[0], y, label="Drawn from posterior")
    # plt.plot(times, res_y, label="Residual data")
    # plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Flux [AU]")
    plt.tight_layout()
    plt.savefig(f"results/SolarFlare/120704187_two_step/fits/kernel_{recovery_mode}_generation_comparison.png")
    plt.show()

    from scipy.signal import periodogram

    freqs, powers = periodogram(y, fs=1.)
    plt.loglog(freqs[1:], powers[1:])
    plt.axvline(1/50, color="red")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [AU]")
    plt.show()
