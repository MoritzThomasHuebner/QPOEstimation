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
    parser.add_argument("--injection_mode", default="qpo", choices=["qpo", "white_noise", "red_noise"], type=str)
    parser.add_argument("--likelihood_model", default="gaussian_process", choices=["gaussian_process", "gaussian_process_windowed"], type=str)
    parser.add_argument("--sampling_frequency", default=256, type=int)
    parser.add_argument("--polynomial_max", default=10, type=int)
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
    plot = args.plot
    segment_length = args.segment_length
    outdir = args.outdir
else:
    matplotlib.use('Qt5Agg')
    minimum_id = 2200
    maximum_id = 2300

    sampling_frequency = 256
    polynomial_max = 10
    injection_mode = "qpo"
    likelihood_model = "gaussian_process_windowed"
    plot = True
    segment_length = 1
    outdir = "injection_files"


times = np.linspace(0, segment_length, int(sampling_frequency * segment_length))
times -= times[0]
times -= times[-1] / 2

min_log_a = -2
max_log_a = 1
min_log_c = -1

band_minimum = 5
band_maximum = 64

priors = bilby.core.prior.PriorDict()
mean_priors = get_polynomial_prior(polynomial_max=polynomial_max)
priors.update(mean_priors)

kernel_priors = get_kernel_prior(kernel_type=injection_mode, min_log_a=min_log_a, max_log_a=max_log_a, min_log_c=min_log_c, band_minimum=band_minimum, band_maximum=band_maximum)
priors.update(kernel_priors)
kernel = get_kernel(kernel_type=injection_mode)



if likelihood_model == "gaussian_process_windowed":
    # priors['window_minimum'] = bilby.core.prior.Beta(minimum=t[0], maximum=t[-1], alpha=1, beta=2, name='window_minimum')
    # priors['window_maximum'] = MinimumPrior(minimum=t[0], maximum=t[-1], order=1, reference_name='window_minimum', name='window_maximum', minimum_spacing=0.5)
    # priors['window_maximum'] = bilby.core.prior.Uniform(minimum=t[0], maximum=t[-1], name='window_maximum')
    priors['window_minimum'] = bilby.core.prior.Uniform(minimum=times[0], maximum=times[0]+0.3, name='window_minimum')
    priors['window_size'] = bilby.core.prior.Uniform(minimum=0.3, maximum=0.7, name='window_size')
    priors['window_maximum'] = bilby.core.prior.Constraint(minimum=-1000, maximum=times[-1], name='window_maximum')

    def window_conversion_func(sample):
        sample['window_maximum'] = sample['window_minimum'] + sample['window_size']
        if injection_mode in ['qpo', 'zeroed_qpo', 'mixed', 'zeroed_mixed']:
            sample = decay_constrain_conversion_function(sample=sample)
        return sample

    if injection_mode in ['qpo', 'zeroed_qpo', 'mixed', 'zeroed_mixed']:
        priors.conversion_function = window_conversion_func
else:
    if injection_mode in ['qpo', 'zeroed_qpo', 'mixed', 'zeroed_mixed']:
        priors.conversion_function = decay_constrain_conversion_function




for injection_id in range(minimum_id, maximum_id):
    params = priors.sample()
    # while np.isinf(priors.ln_prob(params)):
    #     params = priors.sample()
    Path(f'injection_files/{injection_mode}').mkdir(exist_ok=True, parents=True)
    create_injection(params=params, injection_mode=injection_mode, times=times, outdir=outdir,
                     injection_id=injection_id, plot=plot, likelihood_model=likelihood_model)

#
#
# params['mean:a0'] = 0
# params['mean:a1'] = 0
# params['mean:a2'] = 0
# params['mean:a3'] = 0
# params['mean:a4'] = 0

# log_as = np.linspace(maximum_log_a, maximum_log_a, 10)
# log_cs = np.linspace(minimum_log_c, minimum_log_c, 10)
# log_fs = [np.log(20)] * 10
#
# for injection_id in range(minimum_id, maximum_id):
#     bilby.core.utils.logger.info(f"ID: {injection_id}")
#     log_a = log_as[int(str(injection_id%1000).zfill(3)[1])]
#     log_c = log_cs[int(str(injection_id%1000).zfill(3)[2])]
    # log_f = log_fs[int(str(injection_id).zfill(3)[0])]
    # log_f = np.log(20)
    #
    # params['kernel:log_a'] = log_a
    # params['kernel:log_c'] = log_c
    #
    # if injection_mode == "qpo":
    #     params['kernel:log_b'] = -10
    #     params['kernel:log_f'] = log_f
    #
