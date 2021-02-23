import argparse
import json
import os
import sys
from pathlib import Path
import stingray
import bilby
import matplotlib
import matplotlib.pyplot as plt

import QPOEstimation
from QPOEstimation.likelihood import CeleriteLikelihood, WhittleLikelihood, \
    GrothLikelihood, WindowedCeleriteLikelihood, get_kernel, get_mean_model
from QPOEstimation.model.celerite import PolynomialMeanModel
from QPOEstimation.model.series import *
from QPOEstimation.prior.gp import *
from QPOEstimation.prior.mean import get_polynomial_prior
from QPOEstimation.stabilisation import bar_lev
from QPOEstimation.get_data import *

likelihood_models = ["gaussian_process", "gaussian_process_windowed", "periodogram", "poisson"]
modes = ["qpo", "white_noise", "red_noise", "pure_qpo", "general_qpo"]
run_modes = ['select_time', 'sliding_window', 'multiple_windows', 'candidates', 'injection']
background_models = ["polynomial", "exponential", "mean"]
data_modes = ['normal', 'smoothed', 'smoothed_residual', 'blind_injection']


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", default='sliding_window', choices=run_modes)
    parser.add_argument("--sampling_frequency", default=None, type=int)
    parser.add_argument("--data_mode", choices=data_modes, default='normal', type=str)
    parser.add_argument("--alpha", default=0.02, type=float)

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
    parser.add_argument("--periodogram_likelihood", default="whittle", choices=["whittle", "groth"])
    parser.add_argument("--periodogram_noise_model", default="red_noise", choices=["red_noise", "broken_power_law"])

    parser.add_argument("--band_minimum", default=10, type=int)
    parser.add_argument("--band_maximum", default=32, type=int)

    parser.add_argument("--segment_length", default=1.0, type=float)
    parser.add_argument("--segment_step", default=0.27, type=float)
    parser.add_argument("--nlive", default=150, type=int)
    parser.add_argument("--use_ratio", default='False', type=str)

    parser.add_argument("--try_load", default='True', type=str)
    parser.add_argument("--resume", default='False', type=str)
    parser.add_argument("--plot", default='True', type=str)
    args = parser.parse_args()

    run_mode = args.run_mode
    sampling_frequency = args.sampling_frequency
    data_mode = args.data_mode
    alpha = args.alpha

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
    periodogram_likelihood = args.periodogram_likelihood
    periodogram_noise_model = args.periodogram_noise_model

    band_minimum = args.band_minimum
    band_maximum = args.band_maximum

    segment_length = args.segment_length
    segment_step = args.segment_step

    nlive = args.nlive
    use_ratio = boolean_string(args.use_ratio)

    try_load = boolean_string(args.try_load)
    resume = boolean_string(args.resume)
    plot = boolean_string(args.plot)
else:
    matplotlib.use('Qt5Agg')

    run_mode = 'sliding_window'
    # run_mode = 'select_time'
    sampling_frequency = 256
    data_mode = 'smoothed_residual'
    # data_mode = 'normal'
    alpha = 0.02

    start_time = 10
    end_time = 400

    period_number = 13
    run_id = 11

    candidate_id = 3

    injection_id = 2201
    injection_mode = "qpo"

    polynomial_max = 1000
    min_log_a = -5
    max_log_a = 5
    min_log_c = -5
    minimum_window_spacing = 0

    recovery_mode = "red_noise"
    likelihood_model = "gaussian_process_windowed"
    # background_model = "polynomial"
    background_model = "mean"
    periodogram_likelihood = "whittle"
    periodogram_noise_model = "red_noise"

    band_minimum = 5
    band_maximum = 64
    # segment_length = 7.56
    # segment_length = 2.268
    segment_length = 2.4
    # segment_length = 2.
    segment_step = 0.23625  # Requires 32 steps
    # segment_step = 0.54   # Requires 14 steps

    nlive = 100
    use_ratio = True

    try_load = True
    resume = False
    plot = True

    suffix = ""

pulse_period = 7.56  # see papers
n_pulse_periods = 47
time_offset = 20.0

band = f'{band_minimum}_{band_maximum}Hz'

if sampling_frequency is None:
    if band_maximum <= 64:
        sampling_frequency = 256
        alpha = 0.02
    elif band_maximum <= 128:
        sampling_frequency = 512
        alpha = 0.01
    else:
        sampling_frequency = 1024


truths = None
if run_mode == 'candidates':
    times, counts = get_candidates_data(
        candidates_file_dir='candidates', band=band, data_mode=data_mode, candidate_id=candidate_id,
        segment_length=segment_length, sampling_frequency=sampling_frequency, alpha=alpha)
    outdir = f"{run_mode}/{band}/{data_mode}/{recovery_mode}/{likelihood_model}"
    label = f"{candidate_id}"
elif run_mode == 'injection':
    times, counts, truths = get_injection_data(
        injection_file_dir='injection_files', injection_mode=injection_mode, recovery_mode=recovery_mode,
        likelihood_model=likelihood_model, injection_id=injection_id)
    outdir = f"{run_mode}/{band}/{injection_mode}_injection/{recovery_mode}_recovery/{likelihood_model}"
    label = f"{str(injection_id).zfill(2)}"
elif run_mode == 'sliding_window':
    times, counts = get_giant_flare_data_from_period(
        data_mode=data_mode, period_number=period_number, run_id=run_id, segment_step=segment_step,
        segment_length=segment_length, sampling_frequency=sampling_frequency, alpha=alpha)
    outdir = f"{run_mode}/{band}/{data_mode}/{recovery_mode}/{likelihood_model}/period_{period_number}"
    label = f'{run_id}'
elif run_mode == 'select_time':
    times, counts = get_giant_flare_data_from_segment(
        start_time=start_time, end_time=end_time, data_mode=data_mode,
        sampling_frequency=sampling_frequency, alpha=alpha)
    outdir = f"{run_mode}/{band}/{data_mode}/{recovery_mode}/{likelihood_model}"
    label = f'{start_time}_{end_time}'
else:
    raise ValueError


# Move center of light curve to 0
times -= times[0]
times -= times[-1] / 2

priors = bilby.core.prior.PriorDict()
if likelihood_model in ["gaussian_process", "gaussian_process_windowed"]:
    if run_mode == 'injection' or data_mode in ['smoothed', 'smoothed_residual']:
        y = counts
        yerr = np.sqrt(counts)
    else:
        y = bar_lev(counts)
        yerr = np.ones(len(counts))
    plt.errorbar(times, y, yerr=np.sqrt(yerr), fmt=".k", capsize=0, label='data')
    plt.show()
    plt.clf()

    mean_model, fit_mean = get_mean_model(model_type=background_model, y=y)

    if background_model == 'polynomial':
        fit_mean = (polynomial_max != 0)
        mean_priors = get_polynomial_prior(polynomial_max=polynomial_max)
        priors.update(mean_priors)
    else:
        fit_mean = False

    kernel_priors = get_kernel_prior(kernel_type=recovery_mode, min_log_a=min_log_a, max_log_a=max_log_a,
                                     min_log_c=min_log_c, band_minimum=band_minimum, band_maximum=band_maximum)
    priors.update(kernel_priors)
    kernel = get_kernel(kernel_type=recovery_mode)

    if likelihood_model == "gaussian_process_windowed":
        window_priors = get_window_priors(times=times)
        priors.update(window_priors)

        if recovery_mode in ['qpo', 'pure_qpo', 'general_qpo']:
            priors.conversion_function = decay_constrain_conversion_function

        likelihood = WindowedCeleriteLikelihood(mean_model=mean_model, kernel=kernel, fit_mean=fit_mean, t=times,
                                                y=y, yerr=yerr)
    else:
        if recovery_mode in ['qpo', 'pure_qpo', 'general_qpo']:
            priors.conversion_function = decay_constrain_conversion_function
        likelihood = CeleriteLikelihood(kernel=kernel, mean_model=mean_model, fit_mean=fit_mean, t=times,
                                        y=y, yerr=yerr)

meta_data = dict(kernel_type=recovery_mode, mean_model=background_model, times=times,
                 y=y, yerr=yerr, likelihood_model=likelihood_model, truths=truths)


result = None
if try_load:
    try:
        result = bilby.result.read_in_result(outdir=f"{outdir}/results", label=label)
    except Exception:
        pass
else:
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                               label=label, sampler='dynesty', nlive=nlive, sample='rwalk',
                               resume=resume, use_ratio=use_ratio, result_class=QPOEstimation.result.GPResult,
                               meta_data=meta_data)

if plot:
    result.plot_all()


# clean up
for extension in ['_checkpoint_run.png', '_checkpoint_stats.png', '_checkpoint_trace.png',  # '_corner.png',
                  '_dynesty.pickle', '_resume.pickle', '_result.json.old', '_samples.dat']:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
