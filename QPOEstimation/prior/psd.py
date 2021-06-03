import bilby
import numpy as np


def get_red_noise_prior():
    prior = bilby.core.prior.PriorDict()
    prior['alpha'] = bilby.core.prior.Uniform(0, 10, name='alpha')
    prior['log_beta'] = bilby.core.prior.Uniform(np.log(1e-6), np.log(1e6), name='log_beta')
    prior['log_sigma'] = bilby.core.prior.Uniform(np.log(1e-5), np.log(1e5), name='log_sigma')
    return prior


def get_qpo_prior(frequencies=None):
    if frequencies is None:
        df = 1
        max_frequency = 512
    else:
        df = frequencies[1] - frequencies[0]
        max_frequency = frequencies[-1]
    prior = bilby.core.prior.PriorDict()
    prior['log_amplitude'] = bilby.core.prior.Uniform(np.log(1e-6), np.log(1e6), name='log_amplitude')
    prior['log_width'] = bilby.core.prior.Uniform(np.log(df/np.pi), np.log(max_frequency), name='log_width')
    prior['log_frequency'] = bilby.core.prior.Uniform(np.log(2*df), np.log(max_frequency), name='log_frequency')
    return prior


def get_broken_power_law_prior():
    prior = bilby.core.prior.PriorDict()
    prior['alpha_1'] = bilby.core.prior.Uniform(0, 10, name='alpha_1')
    prior['alpha_2'] = bilby.core.prior.Uniform(0, 10, name='alpha_2')
    prior['log_delta'] = bilby.core.prior.Uniform(np.log(1e-6), np.log(1e6), name='log_delta')
    prior['rho'] = bilby.core.prior.DeltaFunction(peak=-1)
    prior['log_beta'] = bilby.core.prior.Uniform(np.log(1e-6), np.log(1e6), name='log_beta')
    prior['log_sigma'] = bilby.core.prior.Uniform(np.log(1e-5), np.log(1e5), name='log_sigma')
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
