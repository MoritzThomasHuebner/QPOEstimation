import bilby
import numpy as np


def get_red_noise_prior():
    prior = bilby.core.prior.PriorDict()
    prior['alpha'] = bilby.core.prior.Uniform(0, 10, name='alpha')
    prior['beta'] = bilby.core.prior.LogUniform(1e-6, 10, name='beta')
    prior['sigma'] = bilby.core.prior.LogUniform(1e-5, 1, name='sigma')
    return prior


def get_qpo_prior(frequencies=None):
    if frequencies is None:
        df = 1
        max_frequency = 512
    else:
        df = frequencies[1] - frequencies[0]
        max_frequency = frequencies[-1]
    prior = bilby.core.prior.PriorDict()
    prior['amplitude'] = bilby.core.prior.LogUniform(1e-4, 100, name='amplitude')
    prior['width'] = bilby.core.prior.Uniform(df/np.pi, 100, name='width')
    prior['central_frequency'] = bilby.core.prior.Uniform(1.0, max_frequency, name='central_frequency')
    # prior['offset'] = bilby.core.prior.Uniform(0, 1, name='offset')
    prior['offset'] = bilby.core.prior.DeltaFunction(0, name='offset')
    return prior


def get_broken_power_law_prior():
    prior = bilby.core.prior.PriorDict()
    prior['alpha_1'] = bilby.core.prior.Uniform(0, 10, name='alpha_1')
    prior['alpha_2'] = bilby.core.prior.Uniform(0, 10, name='alpha_2')
    prior['delta'] = bilby.core.prior.LogUniform(1e-4, 100, name='delta')
    prior['rho'] = bilby.core.prior.DeltaFunction(peak=-1)
    prior['beta'] = bilby.core.prior.LogUniform(0.01, 10, name='beta')
    prior['sigma'] = bilby.core.prior.LogUniform(1e-4, 1, name='sigma')
    return prior


def get_full_prior(noise_model='red_noise', frequencies=None):
    if noise_model == 'broken_power_law':
        noise_prior = get_broken_power_law_prior()
    else:
        noise_prior = get_red_noise_prior()
    prior = get_qpo_prior(frequencies=frequencies)
    for key, value in noise_prior.items():
        prior[key] = value
    return prior