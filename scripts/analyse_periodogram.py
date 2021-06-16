import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import periodogram

import QPOEstimation
from QPOEstimation.get_data import *
from QPOEstimation.parse import parse_args
from QPOEstimation.prior.psd import *
from QPOEstimation.utils import *

if len(sys.argv) > 1:
    parser = parse_args()
    args = parser.parse_args()

    data_source = args.data_source
    run_mode = args.run_mode
    sampling_frequency = args.sampling_frequency
    data_mode = args.data_mode
    alpha = args.alpha
    variance_stabilisation = boolean_string(args.variance_stabilisation)

    hares_and_hounds_id = args.hares_and_hounds_id
    hares_and_hounds_round = args.hares_and_hounds_round

    solar_flare_folder = args.solar_flare_folder
    solar_flare_id = args.solar_flare_id
    grb_id = args.grb_id
    grb_binning = args.grb_binning
    grb_detector = args.grb_detector
    grb_energy_band = args.grb_energy_band
    magnetar_label = args.magnetar_label
    magnetar_tag = args.magnetar_tag
    magnetar_bin_size = args.magnetar_bin_size
    magnetar_subtract_t0 = boolean_string(args.magnetar_subtract_t0)
    magnetar_unbarycentred_time = boolean_string(args.magnetar_unbarycentred_time)

    rebin_factor = args.rebin_factor

    start_time = args.start_time
    end_time = args.end_time

    period_number = args.period_number
    run_id = args.run_id

    candidate_id = args.candidate_id

    polynomial_max = args.polynomial_max
    amplitude_min = args.amplitude_min
    amplitude_max = args.amplitude_max
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max
    t_0_min = args.t_0_min
    t_0_max = args.t_0_max
    tau_min = args.tau_min
    tau_max = args.tau_max

    min_log_a = args.min_log_a
    max_log_a = args.max_log_a
    min_log_c = args.min_log_c
    max_log_c = args.max_log_c
    minimum_window_spacing = args.minimum_window_spacing

    injection_id = args.injection_id
    injection_mode = args.injection_mode
    injection_file_dir = args.injection_file_dir
    injection_likelihood_model = args.injection_likelihood_model

    recovery_mode = args.recovery_mode
    likelihood_model = args.likelihood_model

    band_minimum = args.band_minimum
    band_maximum = args.band_maximum

    segment_length = args.segment_length
    segment_step = args.segment_step

    nlive = args.nlive
    sample = args.sample
    use_ratio = boolean_string(args.use_ratio)

    try_load = boolean_string(args.try_load)
    resume = boolean_string(args.resume)
    plot = boolean_string(args.plot)
    suffix = args.suffix
else:
    matplotlib.use('Qt5Agg')

    data_source = "injection"  # "magnetar_flare_binned"
    run_mode = 'select_time'
    sampling_frequency = 256
    data_mode = 'normal'
    alpha = 0.02
    variance_stabilisation = False

    hares_and_hounds_id = "612579"
    hares_and_hounds_round = 'HH2'

    solar_flare_folder = 'goes'
    solar_flare_id = "go1520110128"
    grb_id = "050128"
    grb_binning = "64ms"
    grb_detector = 'swift'
    grb_energy_band = 'all'

    magnetar_label = 'SGR_0501'
    magnetar_tag = '080823478_lcobs'
    magnetar_bin_size = 0.001
    magnetar_subtract_t0 = True
    magnetar_unbarycentred_time = False
    rebin_factor = 1

    start_time = -20
    end_time = 20

    period_number = 14
    run_id = 6

    candidate_id = 5

    injection_id = 0

    polynomial_max = 1000000
    amplitude_min = None
    amplitude_max = None
    # sigma_min = 0.1
    # sigma_max = 10000
    sigma_min = None
    sigma_max = None
    # t_0_min = 1e-3
    # t_0_max = 1000
    t_0_min = None
    t_0_max = None
    tau_min = None
    tau_max = None

    min_log_a = -20
    max_log_a = 15
    # min_log_c = -10
    min_log_c = None
    max_log_c = None
    # max_log_c = 30
    minimum_window_spacing = 0

    injection_mode = "general_qpo"
    injection_file_dir = "injection_files_pop"
    injection_likelihood_model = "whittle"

    recovery_mode = "general_qpo"
    likelihood_model = "whittle"

    band_minimum = None
    band_maximum = None
    segment_length = 3.5
    # segment_step = 0.945  # Requires 8 steps
    segment_step = 0.23625  # Requires 32 steps

    sample = 'rslice'
    nlive = 500
    use_ratio = False

    try_load = False
    resume = False
    plot = True

    # suffix = f"_{n_components}_fred"

band = f'{band_minimum}_{band_maximum}Hz'

truths = None

recovery_mode_str = recovery_mode

times, y, _, outdir, label = get_data(
    data_source=data_source, band=band, data_mode=data_mode, segment_length=segment_length,
    sampling_frequency=sampling_frequency, alpha=alpha, candidates_file_dir='candidates', candidate_id=candidate_id,
    period_number=period_number, run_id=run_id, segment_step=segment_step, start_time=start_time, end_time=end_time,
    run_mode=run_mode, recovery_mode=recovery_mode, recovery_mode_str=recovery_mode_str, likelihood_model=likelihood_model,
    magnetar_label=magnetar_label,  magnetar_tag=magnetar_tag, magnetar_bin_size=magnetar_bin_size,
    magnetar_subtract_t0=magnetar_subtract_t0, magnetar_unbarycentred_time=magnetar_unbarycentred_time,
    rebin_factor=rebin_factor, solar_flare_folder=solar_flare_folder, solar_flare_id=solar_flare_id,
    grb_id=grb_id, grb_binning=grb_binning, grb_detector=grb_detector, grb_energy_band=grb_energy_band,
    injection_file_dir=injection_file_dir, injection_mode=injection_mode, injection_id=injection_id,
    injection_likelihood_model=injection_likelihood_model, hares_and_hounds_id=hares_and_hounds_id,
    hares_and_hounds_round=hares_and_hounds_round
    )



# if data_source in ['grb', 'solar_flare']:
#     pass
# elif data_source in ['hares_and_hounds']:
#     yerr = np.zeros(len(y))
# elif data_source == 'injection':
#     yerr = np.ones(len(y))
# elif variance_stabilisation:
#     y = bar_lev(y)
#     yerr = np.ones(len(y))
# elif data_source == 'test':
#     pass
# else:
#     yerr = np.sqrt(y)
#     yerr[np.where(yerr == 0)[0]] = 1

# Normalization
# if yerr is not None:
#     yerr /= np.mean(y)
# y = (y - np.mean(y))/np.mean(y)

sampling_frequency = 1/(times[1] - times[0])

freqs, powers = periodogram(y, fs=sampling_frequency, window="hann")


if plot:
    plt.plot(times, y, label='data')
    # plt.plot(times, y, label='flux')
    plt.xlabel("time [s]")
    plt.ylabel("counts")
    plt.show()
    plt.clf()

    plt.loglog(freqs[1:], powers[1:])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Power [AU]')
    plt.show()

if recovery_mode == "red_noise":
    priors = get_red_noise_prior()
elif recovery_mode == "general_qpo":
    priors = get_red_noise_prior()
    priors.update(get_qpo_prior(frequencies=freqs, max_log_width=np.log(0.25)))
elif recovery_mode == "broken_power_law":
    priors = get_broken_power_law_prior()
else:
    raise ValueError

frequency_mask = [True] * len(freqs)
frequency_mask = np.array(frequency_mask)
frequency_mask[0] = False
frequency_mask[np.where(freqs < 0.1)] = False

if recovery_mode == "general_qpo":
    noise_model = "red_noise"
else:
    noise_model = recovery_mode

likelihood = QPOEstimation.likelihood.WhittleLikelihood(
    frequencies=freqs, periodogram=powers, frequency_mask=frequency_mask, noise_model=recovery_mode)
meta_data = dict(kernel_type=recovery_mode, times=times,
                 y=y, likelihood_model=likelihood_model, truths=truths)


result = None
if try_load:
    try:
        result = bilby.result.read_in_result(outdir=f"{outdir}/results", label=label)
    except IOError:
        bilby.utils.logger.info("No result file found. Starting from scratch")
if result is None:
    Path(f"{outdir}/results").mkdir(parents=True, exist_ok=True)
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                               label=label, sampler='dynesty', nlive=nlive, sample=sample,
                               resume=resume, use_ratio=use_ratio)
result.plot_corner()

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

if recovery_mode == 'general_qpo':
    max_l_psd_no_qpo = QPOEstimation.model.psd.red_noise(freqs[1:], alpha=max_like_params['alpha'], beta=np.exp(max_like_params['log_beta'])) + np.exp(max_like_params['log_sigma'])
    max_l_psd = max_l_psd_no_qpo + \
                QPOEstimation.model.psd.lorentzian(freqs[1:], amplitude=np.exp(max_like_params['log_amplitude']),
                                                   central_frequency=np.exp(max_like_params['log_frequency']),
                                                   width=np.exp(max_like_params['log_width']))
elif noise_model == 'red_noise':
    max_l_psd = QPOEstimation.model.psd.red_noise(freqs[1:], alpha=max_like_params['alpha'],
                                                  beta=np.exp(max_like_params['log_beta'])) + np.exp(max_like_params['log_sigma'])
    max_l_psd_no_qpo = max_l_psd
elif noise_model == 'broken_power_law':
    max_l_psd = QPOEstimation.model.psd.broken_power_law_noise(freqs[1:], **max_l_bpl_params) + np.exp(max_like_params['log_sigma'])
    max_l_psd_no_qpo = max_l_psd
else:
    raise ValueError
plt.loglog(freqs[1:], max_l_psd, label="Max Like fit")
threshold = -max_l_psd_no_qpo * np.log((1 - 0.9973)/len(freqs[1:]))   # Bonferroni correction
plt.loglog(freqs[1:], threshold, label="3 sigma threshold")
threshold = -max_l_psd_no_qpo * np.log((1 - 0.954)/len(freqs[1:]))   # Bonferroni correction
plt.loglog(freqs[1:], threshold, label="2 sigma threshold")

if recovery_mode == 'general_qpo':
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
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png',
                  '_dynesty.pickle', '_resume.pickle', '_result.json.old', '_samples.dat']:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
