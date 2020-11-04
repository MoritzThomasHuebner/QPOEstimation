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


def create_injection(params, injection_mode, sampling_frequency, segment_length,
                     outdir='injection_files', injection_id=0, plot=False):
    injection_id = str(injection_id).zfill(2)
    Path(f'{outdir}/{injection_mode}').mkdir(exist_ok=True, parents=True)

    if isinstance(params, bilby.core.prior.PriorDict):
        params = params.sample()
    params_mean = dict()
    params_kernel = dict()
    for param, value in params.items():
        if param.startswith('mean:'):
            params_mean[param.replace('mean:', '')] = value
        elif param.startswith('kernel:'):
            params_kernel[param.replace('kernel:', '')] = value

    if injection_mode == "red_noise":
        kernel = ExponentialTerm(**params_kernel)
    elif injection_mode == "qpo":
        params_kernel['log_b'] = -10
        kernel = QPOTerm(**params_kernel)

    mean_model = PolynomialMeanModel(**params_mean)
    t = np.linspace(0, segment_length, sampling_frequency * segment_length)
    dt = t[1] - t[0]
    yerr = np.ones(len(t))
    cov = np.diag(np.ones(len(t)))

    if injection_mode == "white_noise":
        y = np.random.multivariate_normal(mean_model.get_value(t), cov)
        np.savetxt(f'{outdir}/{injection_mode}/{injection_id}_data.txt', np.array([t, y]).T)
        with open(f'{outdir}/{injection_mode}/{injection_id}_params.json', 'w') as f:
            json.dump(params_mean, f)
    else:
        for i in range(len(t)):
            for j in range(len(t)):
                tau = np.abs(i - j) * dt
                cov[i][j] += kernel.get_value(tau=tau)
        y = np.random.multivariate_normal(mean_model.get_value(t), cov)
        np.savetxt(f'{outdir}/{injection_mode}/{injection_id}_data.txt', np.array([t, y]).T)
        with open(f'{outdir}/{injection_mode}/{injection_id}_params.json', 'w') as f:
            json.dump(params, f)

    if plot:
        x = np.linspace(t[0], t[-1], 5000)
        color = "#ff7f0e"
        plt.errorbar(t, y, yerr=np.ones(len(t)), fmt=".k", capsize=0, label='data')
        if injection_mode != 'white_nosie':
            gp = celerite.GP(kernel=kernel, mean=mean_model)
            gp.compute(t, yerr)
            for param, value in params.items():
                gp.set_parameter(param, value)
            pred_mean, pred_var = gp.predict(y, x, return_var=True)
            pred_std = np.sqrt(pred_var)
            plt.plot(x, pred_mean, color=color, label='Prediction')
            plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std,
                             color=color, alpha=0.3, edgecolor="none")

        pred_mean_poly = mean_model.get_value(x)
        plt.plot(x, pred_mean_poly, color='green', label='Mean function')
        plt.legend()
        plt.savefig(f'{outdir}/{injection_mode}/{injection_id}_data.pdf')
        plt.show()
        plt.clf()