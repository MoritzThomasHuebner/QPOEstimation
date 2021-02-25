import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

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

    solar_flare_id = args.solar_flare_id

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
    skewness_min = args.skewness_min
    skewness_max = args.skewness_max
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

    recovery_mode = args.recovery_mode
    likelihood_model = args.likelihood_model
    background_model = args.background_model
    n_components = args.n_components

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
else:
    matplotlib.use('Qt5Agg')

    data_source = 'injection'
    run_mode = 'entire_segment'
    sampling_frequency = 256
    data_mode = 'normal'
    alpha = 0.02
    variance_stabilisation = True

    solar_flare_id = "120704187"

    start_time = 380
    end_time = 800

    period_number = 13
    run_id = 14

    candidate_id = 5

    injection_id = 0
    injection_mode = "pure_qpo"

    polynomial_max = 1000
    amplitude_min = 10
    amplitude_max = 100
    offset_min = -10
    offset_max = 10
    skewness_min = 0.1
    skewness_max = 10
    sigma_min = 0.1
    sigma_max = 10
    t_0_min = 0
    t_0_max = 3
    tau_min = -10
    tau_max = 10

    min_log_a = -1
    max_log_a = 1
    min_log_c = -1
    # max_log_c = np.nan
    max_log_c = 1
    minimum_window_spacing = 0

    recovery_mode = "pure_qpo"
    likelihood_model = "gaussian_process"
    background_model = "fred"
    n_components = 1

    band_minimum = 1
    band_maximum = 64
    segment_length = 3.0
    segment_step = 0.23625  # Requires 32 steps

    sample = 'rslice'
    nlive = 300
    use_ratio = True

    try_load = False
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

band = f'{band_minimum}_{band_maximum}Hz'

if sampling_frequency is None:
    sampling_frequency = 4 * int(np.round(2 ** np.ceil(np.log2(band_maximum))))

truths = None

if data_source == 'giant_flare':
    times, counts = get_giant_flare_data(
        run_mode, band=band, data_mode=data_mode, segment_length=segment_length, sampling_frequency=sampling_frequency,
        alpha=alpha, candidates_file_dir='candidates', candidate_id=candidate_id, period_number=period_number,
        run_id=run_id, segment_step=segment_step, start_time=start_time, end_time=end_time)
    outdir = f"SGR_1806_20/{run_mode}/{band}/{data_mode}/{recovery_mode}/{likelihood_model}"
    if run_mode == 'candidates':
        label = f"{candidate_id}"
    elif run_mode == 'sliding_window':
        label = f'period_{period_number}/{run_id}'
    elif run_mode == 'select_time':
        label = f'{start_time}_{end_time}'
    elif run_mode == 'entire_segment':
        label = 'entire_segment'
    else:
        raise ValueError
elif data_source == 'solar_flare':
    times, counts = get_solar_flare_data(run_mode, solar_flare_id=solar_flare_id,
                                         start_time=start_time, end_time=end_time)
    outdir = f"solar_flare_{solar_flare_id}/{run_mode}/{recovery_mode}/{likelihood_model}"
    if run_mode == 'select_time':
        label = f'{start_time}_{end_time}'
    elif run_mode == 'entire_segment':
        label = 'entire_segment'
    else:
        raise ValueError
elif data_source == 'injection':
    times, counts, truths = get_injection_data(
        injection_file_dir='injection_files', injection_mode=injection_mode, recovery_mode=recovery_mode,
        likelihood_model=likelihood_model, injection_id=injection_id)
    outdir = get_injection_outdir(injection_mode=injection_mode, recovery_mode=recovery_mode,
                                  likelihood_model=likelihood_model)
    label = f"{str(injection_id).zfill(2)}"
else:
    raise ValueError

if data_source == 'injection':
    y = counts
    yerr = np.ones(len(counts))
elif variance_stabilisation:
    y = bar_lev(counts)
    yerr = np.ones(len(counts))
else:
    y = counts
    yerr = np.sqrt(counts)
    yerr[np.where(yerr == 0)[0]] = 1

if plot:
    plt.errorbar(times, y, yerr=np.sqrt(yerr), fmt=".k", capsize=0, label='data')
    plt.show()
    plt.clf()

mean_model, fit_mean = get_mean_model(model_type=background_model, n_components=n_components, y=y)
kernel = get_kernel(kernel_type=recovery_mode)

priors = get_priors(times=times, likelihood_model=likelihood_model, kernel_type=injection_mode,
                    min_log_a=min_log_a, max_log_a=max_log_a, min_log_c=min_log_c,
                    max_log_c=max_log_c, band_minimum=band_minimum, band_maximum=band_maximum,
                    model_type=background_model, polynomial_max=polynomial_max, minimum_spacing=0,
                    n_components=n_components, **mean_prior_bound_dict)

likelihood = get_celerite_likelihood(mean_model=mean_model, kernel=kernel, fit_mean=fit_mean, times=times,
                                     y=y, yerr=yerr, likelihood_model=likelihood_model)
meta_data = dict(kernel_type=recovery_mode, mean_model=background_model, times=times,
                 y=y, yerr=yerr, likelihood_model=likelihood_model, truths=truths, n_components=n_components)


result = None
if try_load:
    try:
        result = QPOEstimation.result.GPResult.from_json(outdir=f"{outdir}/results", label=label)
    except IOError:
        bilby.utils.logger.info("No result file found. Starting from scratch")
if result is None:
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                               label=label, sampler='dynesty', nlive=nlive, sample=sample,
                               resume=resume, use_ratio=use_ratio, result_class=QPOEstimation.result.GPResult,
                               meta_data=meta_data)

if plot:
    result.plot_all()

# clean up
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png',
                  '_dynesty.pickle', '_resume.pickle', '_result.json.old', '_samples.dat']:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
