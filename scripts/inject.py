import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt

import bilby
import celerite
import matplotlib
import numpy as np

from QPOEstimation.likelihood import QPOTerm, ExponentialTerm, ZeroedQPOTerm, get_kernel
from QPOEstimation.model.celerite import PolynomialMeanModel
from QPOEstimation.injection import create_injection
from QPOEstimation.prior.gp import *
from QPOEstimation.prior.minimum import MinimumPrior

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum_id", default=0, type=int)
    parser.add_argument("--maximum_id", default=100, type=int)
    parser.add_argument("--injection_mode", default="qpo", choices=["qpo", "white_noise", "red_noise", "mixed"], type=str)
    parser.add_argument("--likelihood_model", default="gaussian_process",
                        choices=["gaussian_process", "gaussian_process_windowed"], type=str)
    parser.add_argument("--sampling_frequency", default=256, type=int)

    parser.add_argument("--polynomial_max", default=10, type=int)
    parser.add_argument("--min_log_a", default=-1, type=float)
    parser.add_argument("--max_log_a", default=2, type=float)
    parser.add_argument("--min_log_c", default=-1, type=float)
    parser.add_argument("--max_log_c", default=None, type=float)
    parser.add_argument("--band_minimum", default=5, type=float)
    parser.add_argument("--band_maximum", default=64, type=float)

    parser.add_argument("--plot", default=False, type=bool)
    parser.add_argument("--segment_length", default=1.0, type=float)
    parser.add_argument("--outdir", default='injection_files', type=str)
    args = parser.parse_args()
    minimum_id = args.minimum_id
    maximum_id = args.maximum_id
    injection_mode = args.injection_mode
    likelihood_model = args.likelihood_model
    sampling_frequency = args.sampling_frequency

    polynomial_max = args.polynomial_max
    min_log_a = args.min_log_a
    max_log_a = args.max_log_a
    min_log_c = args.min_log_c
    max_log_c = args.max_log_c
    band_minimum = args.band_minimum
    band_maximum = args.band_maximum

    plot = args.plot
    segment_length = args.segment_length
    outdir = args.outdir
else:
    matplotlib.use('Qt5Agg')
    minimum_id = 2200
    maximum_id = 2300

    sampling_frequency = 256

    polynomial_max = 0
    min_log_a = 2
    max_log_a = 2
    min_log_c = 0
    max_log_c = 0
    band_minimum = 20
    band_maximum = 20

    injection_mode = "qpo"
    likelihood_model = "gaussian_process"
    plot = False
    segment_length = 1
    outdir = "injection_files"


times = np.linspace(0, segment_length, int(sampling_frequency * segment_length))
times -= times[0]
times -= times[-1] / 2

priors = bilby.core.prior.ConditionalPriorDict()
mean_priors = get_polynomial_prior(polynomial_max=polynomial_max)
priors.update(mean_priors)

kernel_priors = get_kernel_prior(kernel_type=injection_mode, min_log_a=min_log_a, max_log_a=max_log_a,
                                 min_log_c=min_log_c, max_log_c=max_log_c,
                                 band_minimum=band_minimum, band_maximum=band_maximum)
priors.update(kernel_priors)
kernel = get_kernel(kernel_type=injection_mode)


if likelihood_model == "gaussian_process_windowed":
    priors.update(get_window_priors(times=times))

if injection_mode in ['qpo', 'pure_qpo', 'mixed', 'zeroed_mixed']:
    priors.conversion_function = decay_constrain_conversion_function

for injection_id in range(minimum_id, maximum_id):
    params = priors.sample()
    Path(f'injection_files/{injection_mode}').mkdir(exist_ok=True, parents=True)
    create_injection(params=params, injection_mode=injection_mode, times=times, outdir=outdir,
                     injection_id=injection_id, plot=plot, likelihood_model=likelihood_model)

