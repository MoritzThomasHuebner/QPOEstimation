from astropy.io import fits
import bilby
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from QPOEstimation.stabilisation import anscombe
from QPOEstimation.likelihood import QPOTerm, CeleriteLikelihood
from scipy.signal import periodogram
import celerite
from celerite import terms

fits_data = fits.open('data/GRB200415A/glg_tte_n0_bn200415367_v00.fit')

# ttes = []
# for det in ['n0', 'n1', 'n2', 'n3', 'n5', 'b0']:
#     fits_data = fits.open(f'data/GRB200415A/glg_tte_{det}_bn200415367_v00.fit')
#     data = fits_data[2].data
#     for i in range(len(data)):
#         ttes.append(data[i][0])
# np.savetxt('data/GRB200415A/combined_ttes.txt', ttes)
ttes = np.loadtxt('data/GRB200415A/combined_ttes.txt')

idx = np.where(np.logical_and(ttes > ttes[0] + 137.05, ttes < ttes[0] + 137.25))
cut_ttes = np.array(ttes)[idx]
plt.hist(cut_ttes, bins=250)
plt.semilogy()
plt.show()

histogram, times = np.histogram(cut_ttes, bins=1000)
times = times[:-1]

plt.semilogy(times, histogram)
plt.show()

n_qpos = 1

c = histogram
t = times

stabilised_counts = anscombe(c)
stabilised_variance = np.ones(len(stabilised_counts))

# A non-periodic component
Q = 1.0 / np.sqrt(2.0)
w0 = 3.0
S0 = np.var(stabilised_counts) / (w0 * Q)

kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))
for i in range(n_qpos):
    kernel += QPOTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=-3)

params_dict = kernel.get_parameter_dict()

gp = celerite.GP(kernel, mean=np.mean(stabilised_counts))
gp.compute(t, stabilised_variance)  # You always need to call compute once.

priors = bilby.core.prior.PriorDict()

if n_qpos == 0:
    priors['kernel:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='log_S0')
    priors['kernel:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='log_omega0')
    priors['kernel:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1/np.sqrt(2)), name='log_Q')
elif n_qpos == 1:
    priors['kernel:terms[0]:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='terms[0]:log_S0')
    priors['kernel:terms[0]:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1 / np.sqrt(2)), name='terms[0]:log_Q')
    priors['kernel:terms[0]:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='terms[0]:log_omega0')
    priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[1]:log_a')
    priors['kernel:terms[1]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[1]:log_b')
    priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[1]:log_c')
    priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(2000), maximum=-np.log(5), name='terms[1]:log_P')
    # priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-4.15, maximum=-2.0, name='terms[1]:log_P')
elif n_qpos == 2:
    priors['kernel:terms[0]:log_S0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='terms[0]:log_S0')
    priors['kernel:terms[0]:log_Q'] = bilby.core.prior.DeltaFunction(peak=np.log(1 / np.sqrt(2)), name='terms[0]:log_Q')
    priors['kernel:terms[0]:log_omega0'] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name='terms[0]:log_omega0')
    priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[1]:log_a')
    priors['kernel:terms[1]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[1]:log_b')
    priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[1]:log_c')
    priors['kernel:terms[1]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(2000), maximum=-np.log(5),
                                                               name='terms[1]:log_P')
    priors['kernel:terms[2]:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='terms[2]:log_a')
    priors['kernel:terms[2]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[2]:log_b')
    priors['kernel:terms[2]:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=3.5, name='terms[2]:log_c')
    priors['kernel:terms[2]:log_P'] = bilby.core.prior.Uniform(minimum=-np.log(2000), maximum=-np.log(5),
                                                               name='terms[2]:log_P')

likelihood = CeleriteLikelihood(gp=gp, y=stabilised_counts)
outdir = "grb200415a"
label = f"run_{n_qpos}_qpos"

result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir,
                           label=label, sampler='dynesty', nlive=300, sample='rwalk',
                           resume=False)
result.plot_corner()

for term in [1, 2]:
    try:
        frequency_samples = []
        for i, sample in enumerate(result.posterior.iloc):
            frequency_samples.append(1 / np.exp(sample[f'kernel:terms[{term}]:log_P']))

        plt.hist(frequency_samples, bins="fd", density=True)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('normalised PDF')
        median = np.median(frequency_samples)
        percentiles = np.percentile(frequency_samples, [16, 84])
        plt.title(
            f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
        plt.savefig(f"{outdir}/frequency_posterior_{label}_{term}")
        plt.clf()
    except Exception:
        continue


max_like_params = result.posterior.iloc[-1]
for name, value in max_like_params.items():
    try:
        gp.set_parameter(name=name, value=value)
    except ValueError:
        continue

x = np.linspace(t[0], t[-1], 5000)
pred_mean, pred_var = gp.predict(stabilised_counts, x, return_var=True)
pred_std = np.sqrt(pred_var)
plt.legend()

color = "#ff7f0e"
plt.errorbar(t, stabilised_counts, yerr=np.zeros(len(stabilised_counts)), fmt=".k", capsize=0, label='data')
plt.plot(x, pred_mean, color=color, label='Prediction')
plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
                 edgecolor="none")
plt.xlabel("time [s]")
plt.ylabel("variance stabilised data")
plt.show()
plt.savefig(f"{outdir}/max_like_fit_{label}")
plt.clf()
