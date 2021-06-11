import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import QPOEstimation
from QPOEstimation.prior.gp import *

import sys

matplotlib.use('Qt5Agg')

data_source = 'grb'
run_mode = 'entire_segment'

start_time = -100
end_time = 100
# start_time = int(sys.argv[1])
# end_time = int(sys.argv[2])

recovery_mode = "qpo"
# recovery_mode = sys.argv[3]
noise_model = "red_noise"


sample = 'rslice'
nlive = 400
use_ratio = False

try_load = False
resume = False
plot = True

suffix = ""


# sampling_frequency = 1/0.064

truths = None

# grb_id = sys.argv[4]
# grb_id = "GRB050128"
data_select = "all"

# data = np.loadtxt(f'data/GRBs/{grb_id}/64ms_lc_ascii_swift.txt')
# times = data[:, 0]
# y = data[:, 9]
# yerr = data[:, 10]
data = np.loadtxt(f'data/test_goes_20130512_{data_select}.txt')
times = data[:, 0]
y = data[:, 1]

# y = data[:, 9]
# yerr = data[:, 10]

# data_label = "ultra_padded"
# data = np.loadtxt(f'data/test_data_{data_label}.txt')
# y = data
# if data_label == 'normal':
#     length = 10
# elif data_label == 'padded':
#     length = 50
# elif data_label == "super_padded":
#     length = 250
# elif data_label == "ultra_padded":
#     length = 1000
# times = np.linspace(-length, length, len(y))
# yerr = np.zeros(len(times))

# indices = np.where(np.logical_and(times > start_time, times < end_time))[0]
# times = times[indices]
# y = y[indices]
# yerr = yerr[indices]
sampling_frequency = 1/(times[1] - times[0])


# indices_narrow = np.where(np.logical_and(times > -10, times < 10))[0]
# y_narrow = y[indices_narrow]

# y[np.where(times < -5)] = 0
# y[np.where(times > 5)] = 0


# plt.errorbar(times, y, yerr=yerr, fmt='.k', capsize=0)
plt.plot(times, y)
plt.xlabel("time [s]")
plt.ylabel("Counts/sec/det")
plt.show()

window = 'hann'
from scipy.signal import periodogram

# import stingray
# lc = stingray.Lightcurve(times, y, yerr)
# p = stingray.Powerspectrum(lc=lc)
# freqs = p.freq
# powers = p.power

# freqs, powers = periodogram(y, fs=1/.064, window=("tukey", 0.5))
# plt.step(freqs[1:], powers[1:], label="tukey")
# print(sampling_frequency)
freqs, powers = periodogram(y, fs=sampling_frequency, window=window)
# freqs_narrow, powers_narrow = periodogram(y_narrow, fs=sampling_frequency, window=window)
# plt.xlim(0.02, 10)
plt.loglog()
# plt.step(freqs[1:], powers[1:], label="100s duration")
plt.step(freqs[1:], powers[1:])
# plt.step(freqs_narrow[1:], powers_narrow[1:], label="20s duration")
plt.axvline(1/12.6, color='black', label="Possible QPO frequency")
plt.xlabel('frequency [Hz]')
plt.ylabel('Power [AU]')
plt.legend()
plt.show()

outdir = f'goes_20130512/{data_select}/{recovery_mode}_{noise_model}/'
# outdir = f'grb_periodogram/{grb_id}/{recovery_mode}_{noise_model}_{start_time}_{end_time}/'
# outdir = f'grb_periodogram/{grb_id}/{recovery_mode}_{noise_model}_{data_label}/'
label = f'{window}_window'


frequency_mask = [True] * len(freqs)
frequency_mask = np.array(frequency_mask)
frequency_mask[0] = False

likelihood = QPOEstimation.likelihood.WhittleLikelihood(
    frequencies=freqs, periodogram=powers, frequency_mask=frequency_mask, noise_model=noise_model)
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

# if data_label == 'normal' and recovery_mode == "qpo":
#     priors['log_width'].minimum = -8

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
    max_l_psd_no_qpo = QPOEstimation.model.psd.red_noise(freqs[1:], alpha=max_like_params['alpha'], beta=np.exp(max_like_params['log_beta'])) + np.exp(max_like_params['log_sigma'])
    max_l_psd = max_l_psd_no_qpo + \
                QPOEstimation.model.psd.lorentzian(freqs[1:], amplitude=np.exp(max_like_params['log_amplitude']),
                                                   central_frequency=np.exp(max_like_params['log_frequency']),
                                                   width=np.exp(max_like_params['log_width']))
elif recovery_mode == 'qpo' and noise_model == 'broken_power_law':

    max_l_psd = QPOEstimation.model.psd.broken_power_law_noise(freqs[1:], **max_l_bpl_params) + \
                QPOEstimation.model.psd.lorentzian(freqs[1:], amplitude=np.exp(max_like_params['log_amplitude']),
                                                   central_frequency=np.exp(max_like_params['log_frequency']),
                                                   width=np.exp(max_like_params['log_width'])) + np.exp(max_like_params['log_sigma'])
elif noise_model == 'red_noise':
    max_l_psd = QPOEstimation.model.psd.red_noise(freqs[1:], alpha=max_like_params['alpha'],
                                                  beta=np.exp(max_like_params['log_beta'])) + np.exp(max_like_params['log_sigma'])
    max_l_psd_no_qpo = max_l_psd
elif noise_model == 'broken_power_law':
    max_l_psd = QPOEstimation.model.psd.broken_power_law_noise(freqs[1:], **max_l_bpl_params) + np.exp(max_like_params['log_sigma'])

plt.loglog(freqs[1:], max_l_psd, label="Max Like fit")
threshold = -max_l_psd_no_qpo * np.log((1 - 0.9973)/len(freqs))   # Bonferroni correction
plt.loglog(freqs[1:], threshold, label="3 sigma threshold")
threshold = -max_l_psd_no_qpo * np.log((1 - 0.954)/len(freqs))   # Bonferroni correction
plt.loglog(freqs[1:], threshold, label="2 sigma threshold")

if recovery_mode == 'qpo':
    max_like_lorentz = QPOEstimation.model.psd.lorentzian(
        freqs[1:], amplitude=np.exp(max_like_params['log_amplitude']),
        central_frequency=np.exp(max_like_params['log_frequency']),
        width=np.exp(max_like_params['log_width']))
    plt.loglog(freqs[1:], max_like_lorentz, label="Max like QPO")
plt.xlabel("frequency [Hz]")
plt.ylabel("Power [AU]")
plt.legend()
plt.savefig(f'{outdir}/{label}_max_like_fit.png')
plt.show()

# clean up
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png', '_checkpoint_trace_unit.png',
                  '_dynesty.pickle', '_resume.pickle', '_result.json.old', '_samples.dat']:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
