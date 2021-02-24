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
from QPOEstimation.prior import get_priors


if len(sys.argv) > 1:
    parser = parse_args()
    parser.add_argument("--minimum_id", default=-1, type=int)
    parser.add_argument("--maximum_id", default=-1, type=int)
    args = parser.parse_args()

    injection_id = args.injection_id
    minimum_id = args.minimum_id
    maximum_id = args.maximum_id
    injection_mode = args.injection_mode

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

    likelihood_model = args.likelihood_model
    background_model = args.background_model
    n_components = args.n_components

    segment_length = args.segment_length
    sampling_frequency = args.sampling_frequency
    band_minimum = args.band_minimum
    band_maximum = args.band_maximum

    plot = args.plot
else:
    matplotlib.use('Qt5Agg')

    injection_id = 0
    minimum_id = -1
    maximum_id = -1
    injection_mode = "qpo"

    polynomial_max = 1000

    amplitude_min = 10
    amplitude_max = 100
    offset_min = -10
    offset_max = 10
    skewness_min = 0.1
    skewness_max = 10
    sigma_min = 0.1
    sigma_max = 1
    t_0_min = 0
    t_0_max = 1
    tau_min = 0
    tau_max = 3

    min_log_a = -1
    max_log_a = 1
    min_log_c = -1
    max_log_c = 1
    minimum_window_spacing = 0

    likelihood_model = "gaussian_process"
    background_model = "fred"
    n_components = 1

    segment_length = 3
    sampling_frequency = 256
    band_minimum = 1
    band_maximum = 64

    plot = True

mean_prior_bounds_dict = dict(
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

times = np.linspace(0, segment_length, int(sampling_frequency * segment_length))

kernel = get_kernel(kernel_type=injection_mode)

priors = get_priors(times=times, likelihood_model=likelihood_model, kernel_type=injection_mode,
                    min_log_a=min_log_a, max_log_a=max_log_a, min_log_c=min_log_c,
                    max_log_c=max_log_c, band_minimum=band_minimum, band_maximum=band_maximum,
                    model_type=background_model, polynomial_max=polynomial_max, minimum_spacing=0,
                    n_components=n_components, **mean_prior_bounds_dict)

params = priors.sample()
outdir = f'injection_files'
Path(outdir).mkdir(exist_ok=True, parents=True)

if minimum_id != maximum_id:
    for injection_id in range(minimum_id, maximum_id):
        create_injection(params=params, injection_mode=injection_mode, times=times, outdir=outdir,
                         injection_id=injection_id, plot=plot, likelihood_model=likelihood_model,
                         mean_model=background_model, n_components=n_components,  poisson_data=False)
else:
    create_injection(params=params, injection_mode=injection_mode, times=times, outdir=outdir,
                     injection_id=injection_id, plot=plot, likelihood_model=likelihood_model,
                     mean_model=background_model, n_components=n_components, poisson_data=False)
