import sys
from pathlib import Path
import numpy as np
import matplotlib

import bilby

from QPOEstimation.injection import create_injection
from QPOEstimation.likelihood import get_kernel
from QPOEstimation.parse import parse_args
from QPOEstimation.prior.gp import get_kernel_prior, get_window_priors
from QPOEstimation.prior.mean import get_mean_prior
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
    min_log_a = args.min_log_a
    max_log_a = args.max_log_a
    min_log_c = args.min_log_c
    max_log_c = args.max_log_c
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
    max_log_c = 1
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


times = np.linspace(0, segment_length, int(sampling_frequency * segment_length))

priors = bilby.core.prior.ConditionalPriorDict()
kernel = get_kernel(kernel_type=injection_mode)
mean_priors = get_mean_prior(model_type=background_model, polynomial_max=polynomial_max, t_min=times[0],
                             t_max=times[-1], minimum_spacing=0, n_components=0)
kernel_priors = get_kernel_prior(
    kernel_type=injection_mode, min_log_a=min_log_a, max_log_a=max_log_a, min_log_c=min_log_c,
    max_log_c=max_log_c, band_minimum=band_minimum, band_maximum=band_maximum)

window_priors = get_window_priors(times=times, likelihood_model=likelihood_model)
priors.update(mean_priors)
priors.update(kernel_priors)
priors.update(window_priors)
priors._resolve_conditions()

params = priors.sample()
outdir = f'injection_files/{injection_mode}/{likelihood_model}'
Path(outdir).mkdir(exist_ok=True, parents=True)

create_injection(params=params, injection_mode=injection_mode, times=times, outdir=outdir, injection_id=injection_id,
                 plot=plot, likelihood_model=likelihood_model, mean_model=background_model,
                 n_components=n_components,  poisson_data=False)
