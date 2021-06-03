import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import QPOEstimation
from QPOEstimation.prior.gp import *

import sys

matplotlib.use('Qt5Agg')

data_source = 'grb'
run_mode = 'entire_segment'

start_time = int(sys.argv[1])
end_time = int(sys.argv[2])

offset = False
polynomial_max = 1000000
amplitude_min = 1e-3
amplitude_max = 1e6
offset_min = 0
offset_max = 10000
skewness_min = 0.1
skewness_max = 10000
sigma_min = 0.1
sigma_max = 10000
t_0_min = start_time - 200
# t_0_min = None
t_0_max = None
tau_min = -10
tau_max = 10

min_log_a = -30
max_log_a = 30
min_log_c = None
# min_log_c = -30
# max_log_c = np.nan
max_log_c = 30
minimum_window_spacing = 0

# recovery_mode = "qpo"
recovery_mode = sys.argv[3]
noise_model = "red_noise"


sample = 'rwalk'
nlive = 400
use_ratio = False

try_load = True
resume = False
plot = True

suffix = ""

mean_prior_bound_dict = dict(
    amplitude_min=amplitude_min,
    amplitude_max=amplitude_max,
    offset_min=offset_min,
    offset_max=offset_max,
    skewness_min=skewness_min,
    skewness_max=skewness_max,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    t_0_min=t_0_min,
    t_0_max=t_0_max,
    tau_min=tau_min,
    tau_max=tau_max
)


sampling_frequency = 1/0.064

truths = None

grb_id = sys.argv[4]

data = np.loadtxt(f'data/GRBs/{grb_id}/64ms_lc_ascii_swift.txt')
times = data[:, 0]
y = data[:, 9]
yerr = data[:, 10]

indices = np.where(np.logical_and(times > start_time, times < end_time))[0]
times = times[indices]
y = y[indices]
yerr = yerr[indices]
plt.errorbar(times, y, yerr=yerr, fmt='.k', capsize=0)
plt.show()
# assert False
window = 'hann'
from scipy.signal import periodogram

# import stingray
# lc = stingray.Lightcurve(times, y, yerr)
# p = stingray.Powerspectrum(lc=lc)
# freqs = p.freq
# powers = p.power

# freqs, powers = periodogram(y, fs=1/.064, window=(window, 0.95))
# print(sampling_frequency)
freqs, powers = periodogram(y, fs=sampling_frequency, window=window)
# plt.xlim(1, 128)
plt.loglog(freqs[1:], powers[1:])
plt.xlabel('frequency [Hz]')
plt.ylabel('Power [AU]')
plt.show()

outdir = f'grb_periodogram/{grb_id}/{recovery_mode}_{noise_model}_{start_time}_{end_time}/'
label = f'{window}_window'





frequency_mask = [True] * len(freqs)
frequency_mask[0] = False
likelihood = QPOEstimation.likelihood.WhittleLikelihood(frequencies=freqs, periodogram=powers,
                                                        frequency_mask=frequency_mask, noise_model=noise_model)
priors = bilby.core.prior.PriorDict()
if noise_model == 'red_noise':
    red_noise_priors = QPOEstimation.prior.psd.get_red_noise_prior()
    priors.update(red_noise_priors)
else:
    broken_power_law_priors = QPOEstimation.prior.psd.get_broken_power_law_prior()
    priors.update(broken_power_law_priors)

if recovery_mode == 'qpo':
    label += '_qpo'
    qpo_priors = QPOEstimation.prior.psd.get_qpo_prior(frequencies=freqs)
    priors.update(qpo_priors)

result = None
if try_load:
    try:
        result = QPOEstimation.result.GPResult.from_json(outdir=f"{outdir}/results", label=label)
    except IOError:
        bilby.utils.logger.info("No result file found. Starting from scratch")
if result is None:
    Path(f"{outdir}/results").mkdir(parents=True, exist_ok=True)
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                               label=label, sampler='dynesty', nlive=nlive, sample=sample,
                               resume=resume, use_ratio=use_ratio)

if plot:
    result.plot_corner()
    # result.plot_lightcurve(end_time=times[-1] + 200)
max_like_params = result.posterior.iloc[-1]
plt.loglog(freqs[1:], powers[1:])

if noise_model == 'broken_power_law':
    names_bpl = ['alpha_1', 'alpha_2', 'log_beta', 'log_delta', 'rho']
    max_l_bpl_params = dict()
    for name in names_bpl:
        max_l_bpl_params[name] = max_like_params[name]
    max_l_bpl_params['beta'] = np.exp(max_l_bpl_params['log_beta'])
    max_l_bpl_params['delta'] = np.exp(max_l_bpl_params['log_delta'])
    del max_l_bpl_params['log_delta']
    del max_l_bpl_params['log_beta']

if recovery_mode == 'qpo' and noise_model == 'red_noise':
    max_l_psd = QPOEstimation.model.psd.red_noise(freqs[1:], alpha=max_like_params['alpha'], beta=np.exp(max_like_params['log_beta'])) + \
                QPOEstimation.model.psd.lorentzian(freqs[1:], amplitude=np.exp(max_like_params['log_amplitude']),
                                                   central_frequency=np.exp(max_like_params['log_frequency']),
                                                   width=np.exp(max_like_params['log_width'])) + \
                np.exp(max_like_params['log_sigma'])

elif recovery_mode == 'qpo' and noise_model == 'broken_power_law':

    max_l_psd = QPOEstimation.model.psd.broken_power_law_noise(freqs[1:], **max_l_bpl_params) + \
                QPOEstimation.model.psd.lorentzian(freqs[1:], amplitude=np.exp(max_like_params['log_amplitude']),
                                                   central_frequency=np.exp(max_like_params['log_frequency']),
                                                   width=np.exp(max_like_params['log_width'])) + np.exp(max_like_params['log_sigma'])
elif noise_model == 'red_noise':
    max_l_psd = QPOEstimation.model.psd.red_noise(freqs[1:], alpha=max_like_params['alpha'],
                                                  beta=np.exp(max_like_params['log_beta'])) + np.exp(max_like_params['log_sigma'])
elif noise_model == 'broken_power_law':
    max_l_psd = QPOEstimation.model.psd.broken_power_law_noise(freqs[1:], **max_l_bpl_params) + np.exp(max_like_params['log_sigma'])

plt.loglog(freqs[1:], max_l_psd)

if recovery_mode == 'qpo':
    max_like_lorentz = QPOEstimation.model.psd.lorentzian(freqs[1:], amplitude=np.exp(max_like_params['log_amplitude']),
                                                          central_frequency=np.exp(max_like_params['log_frequency']),
                                                          width=np.exp(max_like_params['log_width']))
    plt.loglog(freqs[1:], max_like_lorentz)
plt.savefig(f'{outdir}/{label}_max_like_fit.png')
plt.show()

# clean up
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png', '_checkpoint_trace_unit.png',
                  '_dynesty.pickle', '_resume.pickle', '_result.json.old', '_samples.dat']:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
