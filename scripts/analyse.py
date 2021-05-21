import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import QPOEstimation
from QPOEstimation.get_data import *
from QPOEstimation.likelihood import get_kernel, get_mean_model, get_celerite_likelihood
from QPOEstimation.parse import parse_args
from QPOEstimation.prior.gp import *
from QPOEstimation.prior import get_priors
from QPOEstimation.stabilisation import bar_lev
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

    solar_flare_id = args.solar_flare_id
    grb_id = args.grb_id
    grb_binning = args.grb_binning
    grb_detector = args.grb_detector
    magnetar_label = args.magnetar_label
    magnetar_tag = args.magnetar_tag
    magnetar_bin_size = args.magnetar_bin_size
    magnetar_subtract_t0 = boolean_string(args.magnetar_subtract_t0)
    magnetar_unbarycentred_time = boolean_string(args.magnetar_unbarycentred_time)

    start_time = args.start_time
    end_time = args.end_time

    period_number = args.period_number
    run_id = args.run_id

    candidate_id = args.candidate_id

    polynomial_max = args.polynomial_max
    amplitude_min = args.amplitude_min
    amplitude_max = args.amplitude_max
    offset_min = args.offset_min
    offset_max = args.offset_max
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max
    t_0_min = args.t_0_min
    t_0_max = args.t_0_max
    tau_min = args.tau_min
    tau_max = args.tau_max
    offset = boolean_string(args.offset)

    min_log_a = args.min_log_a
    max_log_a = args.max_log_a
    min_log_c = args.min_log_c
    max_log_c = args.max_log_c
    minimum_window_spacing = args.minimum_window_spacing

    injection_id = args.injection_id
    injection_mode = args.injection_mode

    recovery_mode = args.recovery_mode
    likelihood_model = args.likelihood_model
    background_model = args.background_model
    n_components = args.n_components
    jitter_term = boolean_string(args.jitter_term)

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

    data_source = 'hares_and_hounds'
    run_mode = 'entire_segment'
    sampling_frequency = 256
    data_mode = 'normal'
    alpha = 0.02
    variance_stabilisation = False

    hares_and_hounds_id = "612579"
    hares_and_hounds_round = 'HH2'

    solar_flare_id = "go1520110128"
    grb_id = "090709A"
    grb_binning = "1s"
    grb_detector = 'swift'

    magnetar_label = 'SGR_1806_20'
    magnetar_tag = '10223-01-03-01_90931418.874'
    magnetar_bin_size = 0.001
    magnetar_subtract_t0 = True
    magnetar_unbarycentred_time = False

    start_time = -4
    end_time = 103

    period_number = 14
    run_id = 6

    candidate_id = 5

    injection_id = 0

    offset = True
    polynomial_max = 1000000
    amplitude_min = None
    amplitude_max = None
    offset_min = None
    offset_max = None
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

    min_log_a = -10
    max_log_a = 10
    # min_log_c = -10
    min_log_c = None
    max_log_c = None
    # max_log_c = 30
    minimum_window_spacing = 0

    injection_mode = "qpo"
    recovery_mode = "general_qpo"
    likelihood_model = "gaussian_process"
    background_model = "fred_norris_extended"
    n_components = 1
    jitter_term = True

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
if variance_stabilisation:
    suffix = f"_variance_stabilised"
else:
    suffix = ""
suffix += f"_{n_components}_{background_model}s"

mean_prior_bound_dict = dict(
    amplitude_min=amplitude_min,
    amplitude_max=amplitude_max,
    offset_min=offset_min,
    offset_max=offset_max,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    t_0_min=t_0_min,
    t_0_max=t_0_max,
    tau_min=tau_min,
    tau_max=tau_max
)

band = f'{band_minimum}_{band_maximum}Hz'

if sampling_frequency is None:
    sampling_frequency = 4 * int(np.round(2 ** np.ceil(np.log2(band_maximum))))

truths = None

if data_source == 'giant_flare':
    times, counts = get_giant_flare_data(
        run_mode, band=band, data_mode=data_mode, segment_length=segment_length, sampling_frequency=sampling_frequency,
        alpha=alpha, candidates_file_dir='candidates', candidate_id=candidate_id, period_number=period_number,
        run_id=run_id, segment_step=segment_step, start_time=start_time, end_time=end_time)
    outdir = f"SGR_1806_20/{run_mode}/{band}/{recovery_mode}/{likelihood_model}/"
    if run_mode == 'candidates':
        label = f"{candidate_id}"
    elif run_mode == 'sliding_window':
        outdir += f'period_{period_number}/'
        label = f'{run_id}'
    elif run_mode == 'select_time':
        label = f'{start_time}_{end_time}'
    elif run_mode == 'entire_segment':
        label = 'entire_segment'
    else:
        raise ValueError
elif data_source == 'magnetar_flare':
    times, counts = get_tte_magnetar_flare_data(
        run_mode=run_mode, magnetar_label=magnetar_label, tag=magnetar_tag, bin_size=magnetar_bin_size,
        subtract_t0=magnetar_subtract_t0, unbarycentred_time=magnetar_unbarycentred_time, start_time=start_time,
        end_time=end_time)
    outdir = f"magnetar_flares/{magnetar_label}/{magnetar_tag}/{run_mode}/{recovery_mode}/{likelihood_model}/"
    if run_mode == 'select_time':
        label = f'{start_time}_{end_time}'
    else:
        label = run_mode
elif data_source == 'solar_flare':
    times, y, yerr = get_solar_flare_data(run_mode, solar_flare_id=solar_flare_id,
                                          start_time=start_time, end_time=end_time)
    outdir = f"solar_flare_{solar_flare_id}/{run_mode}/{recovery_mode}/{likelihood_model}"
    if run_mode == 'select_time':
        label = f'{start_time}_{end_time}'
    else:
        label = run_mode
elif data_source == 'grb':
    times, y, yerr = get_grb_data(
        run_mode, grb_id=grb_id, grb_binning=grb_binning,
        start_time=start_time, end_time=end_time, grb_detector=grb_detector)
    outdir = f"GRB{grb_id}_{grb_detector}/{run_mode}/{recovery_mode}/{likelihood_model}"
    if run_mode == 'select_time':
        label = f'{start_time}_{end_time}'
    else:
        label = run_mode
    times -= times[0]

elif data_source == 'injection':
    times, counts, truths = get_injection_data(
        injection_file_dir='injection_files', injection_mode=injection_mode, recovery_mode=recovery_mode,
        likelihood_model=likelihood_model, injection_id=injection_id)
    outdir = get_injection_outdir(injection_mode=injection_mode, recovery_mode=recovery_mode,
                                  likelihood_model=likelihood_model)
    label = f"{str(injection_id).zfill(2)}"
elif data_source == 'hares_and_hounds':
    times, y = get_hares_and_hounds_data(run_mode, hares_and_hounds_id=hares_and_hounds_id,
                                         hares_and_hounds_round=hares_and_hounds_round,
                                         start_time=start_time, end_time=end_time)
    times -= times[0]
    outdir = f"hares_and_hounds_{hares_and_hounds_round}/{hares_and_hounds_id}/{run_mode}/{recovery_mode}/{likelihood_model}"
    if run_mode == 'select_time':
        label = f'{start_time}_{end_time}'
    else:
        label = run_mode
else:
    raise ValueError
# from scipy.signal import periodogram
# freqs, powers = periodogram(counts, fs=1)
# # plt.xlim(1, 128)
# plt.loglog(freqs[1:], powers[1:])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('Power [AU]')
# plt.show()

if data_source in ['grb', 'solar_flare']:
    pass
elif data_source == 'hares_and_hounds':
    yerr = np.zeros(len(y))
elif data_source == 'injection':
    y = counts
    yerr = np.ones(len(counts))
elif variance_stabilisation:
    y = bar_lev(counts)
    yerr = np.ones(len(counts))
else:
    y = counts
    yerr = np.sqrt(counts)
    yerr[np.where(yerr == 0)[0]] = 1

# y /= 1000
# yerr /= 1000


if plot:
    plt.errorbar(times, y, yerr=yerr, fmt=".k", capsize=0, label='data')
    # plt.plot(times, y, label='flux')
    plt.xlabel("time [s]")
    plt.ylabel("counts")
    plt.show()
    plt.clf()

mean_model, fit_mean = get_mean_model(model_type=background_model, n_components=n_components, y=y, offset=offset)

priors = get_priors(times=times, y=y, likelihood_model=likelihood_model, kernel_type=recovery_mode,
                    min_log_a=min_log_a, max_log_a=max_log_a, min_log_c=min_log_c,
                    max_log_c=max_log_c, band_minimum=band_minimum, band_maximum=band_maximum,
                    model_type=background_model, polynomial_max=polynomial_max, minimum_spacing=0,
                    n_components=n_components, offset=offset, jitter_term=jitter_term, **mean_prior_bound_dict)

kernel = get_kernel(kernel_type=recovery_mode, jitter_term=jitter_term)
likelihood = get_celerite_likelihood(mean_model=mean_model, kernel=kernel, fit_mean=fit_mean, times=times,
                                     y=y, yerr=yerr, likelihood_model=likelihood_model)
# likelihood = bilby.likelihood.ZeroLikelihood(likelihood)
meta_data = dict(kernel_type=recovery_mode, mean_model=background_model, times=times,
                 y=y, yerr=yerr, likelihood_model=likelihood_model, truths=truths, n_components=n_components,
                 offset=offset, jitter_term=jitter_term)


label += suffix
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
                               resume=resume, use_ratio=use_ratio, result_class=QPOEstimation.result.GPResult,
                               meta_data=meta_data, save=True, gzip=False)

if plot:
    result.plot_all()
    result.plot_lightcurve(end_time=times[-1] + (times[-1] - times[0]) * 0.2)

# clean up
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png',
                  '_dynesty.pickle', '_resume.pickle', '_result.json.old', '_samples.dat']:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
