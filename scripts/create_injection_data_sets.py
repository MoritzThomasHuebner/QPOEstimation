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

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum_id", default=0, type=int)
    parser.add_argument("--maximum_id", default=100, type=int)
    parser.add_argument("--injection_mode", default="one_qpo", choices=["one_qpo", "no_qpo", "red_noise"], type=str)
    parser.add_argument("--sampling_frequency", default=256, type=int)
    parser.add_argument("--polynomial_max", default=10, type=int)
    args = parser.parse_args()
    minimum_id = args.minimum_id
    maximum_id = args.maximum_id
    injection_mode = args.injection_mode
    sampling_frequency = args.sampling_frequency
    polynomial_max = args.polynomial_max
else:
    matplotlib.use('Qt5Agg')
    minimum_id = 0
    maximum_id = 100

    sampling_frequency = 256
    polynomial_max = 10
    injection_mode = "red_noise"


def conversion_function(sample):
    out_sample = deepcopy(sample)
    out_sample['decay_constraint'] = out_sample['kernel:log_c'] - out_sample['kernel:log_f']
    return out_sample



priors = bilby.core.prior.PriorDict()
priors['mean:a0'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a0')
priors['mean:a1'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a1')
priors['mean:a2'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a2')
priors['mean:a3'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a3')
priors['mean:a4'] = bilby.core.prior.Uniform(minimum=-polynomial_max, maximum=polynomial_max, name='mean:a4')
if injection_mode == "red_noise":
    priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=-1, maximum=1, name='log_a')
    priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=1, maximum=np.log(sampling_frequency), name='log_c')
elif injection_mode == "one_qpo":
    priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=-1, maximum=1, name='log_a')
    priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='log_b')
    priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=1, maximum=np.log(sampling_frequency), name='log_c')
    priors['kernel:log_f'] = bilby.core.prior.Uniform(minimum=np.log(5), maximum=np.log(64), name='log_f')
    # priors['kernel:log_f'] = np.log(10)
    priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=-0.5, name='decay_constraint')
    priors.conversion_function = conversion_function
elif injection_mode == "no_qpo":
    priors['kernel:log_a'] = -10
    priors['kernel:log_b'] = 10
    priors['kernel:log_c'] = 0
    priors['kernel:log_f'] = 0

Path(f'injection_files/{injection_mode}').mkdir(exist_ok=True, parents=True)

for injection_id in range(minimum_id, maximum_id):
    params = priors.sample()
    params_mean = dict(a0=params['mean:a0'], a1=params['mean:a1'], a2=params['mean:a2'],
                       a3=params['mean:a3'], a4=params['mean:a4'])
    if injection_mode == "red_noise":
        params_kernel = dict(log_a=params['kernel:log_a'],
                             log_c=params['kernel:log_c'])
        kernel = ExponentialTerm(**params_kernel)
    else:
        params_kernel = dict(log_a=params['kernel:log_a'], log_b=-10,
                             log_c=params['kernel:log_c'], log_f=params['kernel:log_f'])
        kernel = QPOTerm(**params_kernel)

    mean_model = PolynomialMeanModel(**params_mean)
    t = np.linspace(0, 1, sampling_frequency)
    yerr = np.ones(len(t))

    if injection_mode == "no_qpo":
        K = np.diag(np.ones(len(t)))
        y = np.random.multivariate_normal(mean_model.get_value(t), K)
        np.savetxt(f'injection_files/{injection_mode}/{str(injection_id).zfill(2)}_data.txt', np.array([t, y]).T)
        with open(f'injection_files/{injection_mode}/{str(injection_id).zfill(2)}_params.json', 'w') as f:
            json.dump(params_mean, f)

        jitter_kernel = celerite.terms.JitterTerm(log_sigma=-20)
        gp = celerite.GP(kernel=jitter_kernel, mean=mean_model)
        gp.compute(t, yerr)

        x = np.linspace(t[0], t[-1], 5000)
        # pred_mean_poly, pred_var = gp.predict(y, x, return_var=True)
        # pred_std = np.sqrt(pred_var)
    elif injection_mode in ["one_qpo", "red_noise"]:
        K = np.diag(np.ones(len(t)))
        for i in range(len(t)):
            for j in range(len(t)):
                dt = t[1] - t[0]
                tau = np.abs(i - j) * dt
                K[i][j] += kernel.get_value(tau=tau)
        y = np.random.multivariate_normal(mean_model.get_value(t), K)
        np.savetxt(f'injection_files/{injection_mode}/{str(injection_id).zfill(2)}_data.txt', np.array([t, y]).T)
        with open(f'injection_files/{injection_mode}/{str(injection_id).zfill(2)}_params.json', 'w') as f:
            json.dump(params, f)


    # gp = celerite.GP(kernel=kernel, mean=mean_model)
    # gp.compute(t, yerr)
    # for param, value in params.items():
    #     gp.set_parameter(param, value)
    # x = np.linspace(t[0], t[-1], 5000)
    # pred_mean, pred_var = gp.predict(y, x, return_var=True)
    # pred_std = np.sqrt(pred_var)
    #
    # color = "#ff7f0e"
    # plt.errorbar(t, y, yerr=np.ones(len(t)), fmt=".k", capsize=0, label='data')
    # plt.plot(x, pred_mean, color=color, label='Prediction')
    # plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
    #                  edgecolor="none")
    #
    # plt.plot(x, pred_mean_poly, color='green', label='Polynomial')
    # plt.legend()
    # plt.savefig(f'injection_files/{injection_mode}/{str(injection_id).zfill(2)}_data.pdf')
    # plt.show()
    # plt.clf()
