import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

import QPOEstimation
from QPOEstimation.get_data import *
from QPOEstimation.likelihood import get_kernel, get_mean_model, get_celerite_likelihood
from QPOEstimation.model import mean_model_dict
from QPOEstimation.prior.gp import *
from QPOEstimation.prior.mean import get_mean_prior
from QPOEstimation.stabilisation import bar_lev
from QPOEstimation.utils import get_injection_outdir

likelihood_models = ["gaussian_process", "gaussian_process_windowed", "periodogram", "poisson"]
modes = ["qpo", "white_noise", "red_noise", "pure_qpo", "general_qpo"]
data_sources = ['injection', 'giant_flare', 'solar_flare']
run_modes = ['select_time', 'sliding_window', 'candidates', 'entire_segment']
background_models = ["polynomial", "exponential", "fred", "gaussian", "log_normal", "lorentzian", "mean"]
data_modes = ['normal', 'smoothed', 'smoothed_residual', 'blind_injection']


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default='giant_flare', choices=data_sources)
    parser.add_argument("--run_mode", default='sliding_window', choices=run_modes)
    parser.add_argument("--sampling_frequency", default=None, type=int)
    parser.add_argument("--data_mode", choices=data_modes, default='normal', type=str)
    parser.add_argument("--alpha", default=0.02, type=float)
    parser.add_argument("--variance_stabilisation", default='True', type=str)

    parser.add_argument("--solar_flare_id", default='120704187', type=str)

    parser.add_argument("--start_time", default=0., type=float)
    parser.add_argument("--end_time", default=1., type=float)

    parser.add_argument("--period_number", default=0, type=int)
    parser.add_argument("--run_id", default=0, type=int)

    parser.add_argument("--candidate_id", default=0, type=int)

    parser.add_argument("--injection_id", default=0, type=int)
    parser.add_argument("--injection_mode", default="qpo", choices=modes, type=str)

    parser.add_argument("--polynomial_max", default=1000, type=float)
    parser.add_argument("--min_log_a", default=-5, type=float)
    parser.add_argument("--max_log_a", default=15, type=float)
    parser.add_argument("--min_log_c", default=-6, type=float)
    parser.add_argument("--minimum_window_spacing", default=0, type=float)

    parser.add_argument("--recovery_mode", default="qpo", choices=modes)
    parser.add_argument("--model", default="gaussian_process", choices=likelihood_models)
    parser.add_argument("--background_model", default="polynomial", choices=background_models)
    parser.add_argument("--n_components", default=1, type=int)

    parser.add_argument("--band_minimum", default=10, type=int)
    parser.add_argument("--band_maximum", default=32, type=int)

    parser.add_argument("--segment_length", default=1.0, type=float)
    parser.add_argument("--segment_step", default=0.27, type=float)

    parser.add_argument("--nlive", default=150, type=int)
    parser.add_argument("--sample", default='rwalk', type=str)
    parser.add_argument("--use_ratio", default='False', type=str)

    parser.add_argument("--try_load", default='True', type=str)
    parser.add_argument("--resume", default='False', type=str)
    parser.add_argument("--plot", default='True', type=str)
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
    min_log_a = args.min_log_a
    max_log_a = args.max_log_a
    min_log_c = args.min_log_c
    minimum_window_spacing = args.minimum_window_spacing

    injection_id = args.injection_id
    injection_mode = args.injection_mode

    recovery_mode = args.recovery_mode
    likelihood_model = args.model
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

    data_source = 'solar_flare'
    run_mode = 'entire_segment'
    sampling_frequency = 256
    data_mode = 'normal'
    alpha = 0.02
    variance_stabilisation = False

    solar_flare_id = "120704187"

    start_time = 380
    end_time = 800

    period_number = 13
    run_id = 14

    candidate_id = 5

    injection_id = 2201
    injection_mode = "qpo"

    polynomial_max = 1000
    min_log_a = -5
    max_log_a = 25
    min_log_c = -25
    minimum_window_spacing = 0

    recovery_mode = "pure_qpo"
    likelihood_model = "gaussian_process"
    background_model = "gaussian"
    n_components = 3

    band_minimum = 1/400
    band_maximum = 1
    segment_length = 1.0
    segment_step = 0.23625  # Requires 32 steps

    sample = 'rslice'
    nlive = 300
    use_ratio = True

    try_load = False
    resume = False
    plot = True

    suffix = ""

band = f'{band_minimum}_{band_maximum}Hz'

if sampling_frequency is None:
    sampling_frequency = 4*int(np.round(2**np.ceil(np.log2(band_maximum))))

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
    outdir = get_injection_outdir(band=band, injection_mode=injection_mode, recovery_mode=recovery_mode,
                                  likelihood_model=likelihood_model)
    label = f"{str(injection_id).zfill(2)}"
else:
    raise ValueError


if variance_stabilisation:
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


priors = bilby.core.prior.ConditionalPriorDict()
mean_model, fit_mean = get_mean_model(model_type=background_model, n_components=n_components, y=y)
mean_priors = get_mean_prior(model_type=background_model, n_components=n_components, t_min=times[0], t_max=times[-1],
                             minimum_spacing=0, polynomial_max=polynomial_max)

kernel = get_kernel(kernel_type=recovery_mode)
kernel_priors = get_kernel_prior(
    kernel_type=recovery_mode, min_log_a=min_log_a, max_log_a=max_log_a,
    min_log_c=min_log_c, band_minimum=band_minimum, band_maximum=band_maximum)

likelihood = get_celerite_likelihood(mean_model=mean_model, kernel=kernel, fit_mean=fit_mean, times=times,
                                     y=y, yerr=yerr, likelihood_model=likelihood_model)

window_priors = get_window_priors(times=times, likelihood_model=likelihood_model)

meta_data = dict(kernel_type=recovery_mode, mean_model=background_model, times=times,
                 y=y, yerr=yerr, likelihood_model=likelihood_model, truths=truths, n_components=n_components)

priors.update(mean_priors)
priors.update(kernel_priors)
priors.update(window_priors)
priors._resolve_conditions()

result = None
if try_load:
    try:
        result = QPOEstimation.result.GPResult.from_json(outdir=f"{outdir}/results", label=label)
    except IOError:
        bilby.utils.logger.info("No result file found. Starting from scratch")
else:
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
