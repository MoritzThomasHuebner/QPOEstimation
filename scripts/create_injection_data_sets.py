import numpy as np
import bilby
from QPOEstimation.poisson import poisson_process
from QPOEstimation.model.series import sine_gaussian_with_background
from QPOEstimation.likelihood import CeleriteLikelihood
import matplotlib.pyplot as plt
import matplotlib
import bilby
import stingray
from copy import deepcopy
import json
import celerite
from QPOEstimation.model.series import PolynomialMeanModel
from QPOEstimation.likelihood import QPOTerm

matplotlib.use('Qt5Agg')


def conversion_function(sample):
    out_sample = deepcopy(sample)
    out_sample['decay_constraint'] = out_sample['kernel:log_c'] - out_sample['kernel:log_f']
    return out_sample

sampling_frequency = 256

priors = bilby.core.prior.PriorDict()
priors['mean:a0'] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name='mean:a0')
priors['mean:a1'] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name='mean:a1')
priors['mean:a2'] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name='mean:a2')
priors['mean:a3'] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name='mean:a3')
priors['mean:a4'] = bilby.core.prior.Uniform(minimum=-1000, maximum=1000, name='mean:a4')
priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=-5, maximum=15, name='log_a')
priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='log_b')
priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=-6, maximum=np.log(sampling_frequency), name='log_c')
priors['kernel:log_f'] = bilby.core.prior.Uniform(minimum=np.log(10), maximum=np.log(64), name='log_f')
priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=-0.5, name='decay_constraint')
priors.conversion_function = conversion_function

sampling_frequency = 256

for injection_id in range(100):
    res_id = np.random.randint(0, 18, 1)[0]
    res = bilby.result.read_in_result(f'sliding_window_10_40Hz_candidates/one_qpo/results/{res_id}_result.json')
    params = res.posterior.iloc[np.random.randint(len(res.posterior))]
    params_mean = dict(a0=params['mean:a0'], a1=params['mean:a1'], a2=params['mean:a2'],
                       a3=params['mean:a3'], a4=params['mean:a4'])
    params_kernel = dict(log_a=params['kernel:log_a'], log_b=-10,
                         log_c=params['kernel:log_c'], log_f=params['kernel:log_f'])
    params = dict(**params)
    del params['log_likelihood']
    del params['log_prior']
    mean_model = PolynomialMeanModel(**params_mean)
    kernel = QPOTerm(**params_kernel)

    t = np.linspace(0, 1, 256)
    yerr = np.ones(len(t))
    K = np.diag(np.ones(len(t)))
    y = np.random.multivariate_normal(mean_model.get_value(t), K)
    np.savetxt(f'injection_files/no_qpo/{str(injection_id).zfill(2)}_data.txt', np.array([t, y]).T)
    with open(f'injection_files/no_qpo/{str(injection_id).zfill(2)}_params.json', 'w') as f:
        json.dump(params, f)

    for i in range(len(t)):
        for j in range(len(t)):
            dt = t[1] - t[0]
            tau = np.abs(i - j) * dt
            K[i][j] += kernel.get_value(tau=tau)
    y = np.random.multivariate_normal(mean_model.get_value(t), K)
    np.savetxt(f'injection_files/one_qpo/{str(injection_id).zfill(2)}_data.txt', np.array([t, y]).T)
    with open(f'injection_files/one_qpo/{str(injection_id).zfill(2)}_params.json', 'w') as f:
        json.dump(params, f)
