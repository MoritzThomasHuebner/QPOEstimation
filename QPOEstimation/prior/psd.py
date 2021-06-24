import bilby
import numpy as np


def get_red_noise_prior():
    prior = bilby.core.prior.ConditionalPriorDict()
    prior['alpha'] = bilby.core.prior.Uniform(0, 10, name='alpha')
    prior['log_beta'] = bilby.core.prior.Uniform(-60, 60, name='log_beta')
    prior['log_sigma'] = bilby.core.prior.Uniform(-30, 30, name='log_sigma')
    return prior


def log_width_condition_func(reference_params, log_frequency):
    frequency = np.exp(log_frequency)
    width_max = frequency/2
    log_width_max = np.log(width_max)
    return dict(maximum=log_width_max)


def get_qpo_prior(frequencies=None, **kwargs):

    if frequencies is None:
        df = 1
        max_frequency = 512
    else:
        df = frequencies[1] - frequencies[0]
        max_frequency = frequencies[-1]
    prior = bilby.core.prior.ConditionalPriorDict()
    prior['log_amplitude'] = bilby.core.prior.Uniform(-30, 30, name='log_amplitude')
    prior['log_width'] = bilby.core.prior.ConditionalUniform(condition_func=log_width_condition_func,
        minimum=np.log(df/np.pi), maximum=kwargs.get('max_log_width', np.log(0.25*max_frequency)), name='log_width')
    # prior['log_frequency'] = bilby.core.prior.Uniform(np.log(2*df), np.log(max_frequency), name='log_frequency')
    prior['log_frequency'] = bilby.core.prior.Uniform(kwargs.get('min_log_f', np.log(2*df)), np.log(max_frequency), name='log_frequency')
    prior._resolve_conditions()
    return prior


def get_broken_power_law_prior():
    prior = bilby.core.prior.PriorDict()
    prior['alpha_1'] = bilby.core.prior.Uniform(-10, 10, name='alpha_1')
    prior['alpha_2'] = bilby.core.prior.Uniform(-10, 10, name='alpha_2')
    prior['log_delta'] = bilby.core.prior.Uniform(-30, 30, name='log_delta')
    prior['rho'] = bilby.core.prior.DeltaFunction(peak=-1)
    prior['log_beta'] = bilby.core.prior.Uniform(-30, 30, name='log_beta')
    prior['log_sigma'] = bilby.core.prior.Uniform(-30, 30, name='log_sigma')
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
