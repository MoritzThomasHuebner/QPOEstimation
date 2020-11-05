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

from QPOEstimation.likelihood import QPOTerm, ExponentialTerm
from QPOEstimation.model.series import PolynomialMeanModel
from QPOEstimation.injection import create_injection

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum_id", default=0, type=int)
    parser.add_argument("--maximum_id", default=100, type=int)
    parser.add_argument("--injection_mode", default="qpo", choices=["qpo", "white_noise", "red_noise"], type=str)
    parser.add_argument("--sampling_frequency", default=256, type=int)
    parser.add_argument("--polynomial_max", default=10, type=int)
    parser.add_argument("--plot", default=False, type=bool)
    parser.add_argument("--segment_length", default=1.0, type=float)
    parser.add_argument("--outdir", default='injection_files', type=str)
    args = parser.parse_args()
    minimum_id = args.minimum_id
    maximum_id = args.maximum_id
    injection_mode = args.injection_mode
    sampling_frequency = args.sampling_frequency
    polynomial_max = args.polynomial_max
    plot = args.plot
    segment_length = args.segment_length
    outdir = args.outdir
else:
    matplotlib.use('Qt5Agg')
    minimum_id = 0
    maximum_id = 1000

    sampling_frequency = 256
    polynomial_max = 10
    injection_mode = "qpo"
    plot = True
    segment_length = 1
    outdir = 'injection_files'


# def conversion_function(sample):
#     out_sample = deepcopy(sample)
#     out_sample['decay_constraint'] = out_sample['kernel:log_c'] - out_sample['kernel:log_f']
#     return out_sample

# priors = bilby.core.prior.PriorDict()
# priors['mean:a0'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a0')
# priors['mean:a1'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a1')
# priors['mean:a2'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a2')
# priors['mean:a3'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a3')
# priors['mean:a4'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a4')
# if injection_mode == "red_noise":
#     priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=-1, maximum=1, name='log_a')
#     priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=1, maximum=np.log(sampling_frequency), name='log_c')
# elif injection_mode == "one_qpo":
#     priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=-1, maximum=1, name='log_a')
#     priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='log_b')
#     priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=1, maximum=np.log(sampling_frequency), name='log_c')
#     priors['kernel:log_f'] = bilby.core.prior.Uniform(minimum=np.log(5), maximum=np.log(64), name='log_f')
#     # priors['kernel:log_f'] = np.log(10)
#     priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=-0.5, name='decay_constraint')
#     priors.conversion_function = conversion_function
# elif injection_mode == "no_qpo":
#     priors['kernel:log_a'] = -10
#     priors['kernel:log_b'] = 10
#     priors['kernel:log_c'] = 0
#     priors['kernel:log_f'] = 0
#
# Path(f'injection_files/{injection_mode}').mkdir(exist_ok=True, parents=True)
#
# for injection_id in range(minimum_id, maximum_id):
#     create_injection(params=priors.sample(), injection_mode=injection_mode, sampling_frequency=sampling_frequency,
#                      segment_length=segment_length, outdir=outdir, injection_id=injection_id, plot=plot)

params = dict()


params['mean:a0'] = 0
params['mean:a1'] = 0
params['mean:a2'] = 0
params['mean:a3'] = 0
params['mean:a4'] = 0

minimum_log_a = -2
maximum_log_a = 1
minimum_log_c = 1
maximum_log_c = 4.8
minimum_log_f = np.log(10)
maximum_log_f = np.log(64)

log_as = np.linspace(minimum_log_a, maximum_log_a, 10)
log_cs = np.linspace(minimum_log_c, maximum_log_c, 10)
log_fs = np.linspace(minimum_log_f, maximum_log_f, 10)

for injection_id in range(minimum_id, maximum_id):
    bilby.core.utils.logger.info(f"ID: {injection_id}")
    log_a = log_as[int(str(injection_id).zfill(3)[1])]
    log_c = log_cs[int(str(injection_id).zfill(3)[2])]
    log_f = log_fs[int(str(injection_id).zfill(3)[0])]

    params['kernel:log_a'] = log_a
    params['kernel:log_c'] = log_c

    if injection_mode == "qpo":
        params['kernel:log_b'] = -10
        params['kernel:log_f'] = log_f

    Path(f'injection_files/{injection_mode}').mkdir(exist_ok=True, parents=True)
    create_injection(params=params, injection_mode=injection_mode, sampling_frequency=sampling_frequency,
                     segment_length=segment_length, outdir=outdir, injection_id=injection_id, plot=plot)
