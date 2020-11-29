import numpy as np
import matplotlib.pyplot as plt
import bilby
from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import Uniform
from bilby.core.sampler import run_sampler
from bilby.core.result import make_pp_plot
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.hyper.model import Model

recovery_mode = 'mixed'
# outdir = 'testing_hyper_pe_qpo'
outdir = f'testing_hyper_pe_{recovery_mode}'
label = 'gaussian'


band_minimum = 5
band_maximum = 64

results = bilby.result.ResultList([bilby.result.read_in_result(f'sliding_window_{band_minimum}_{band_maximum}Hz_normal/period_{i}/{recovery_mode}/results/27_gaussian_process_result.json') for i in range(46)])
white_noise_results = bilby.result.ResultList([bilby.result.read_in_result(f'sliding_window_{band_minimum}_{band_maximum}Hz_normal/period_{i}/white_noise/results/27_gaussian_process_result.json') for i in range(46)])



# def hyper_prior_log_f(dataset, mu_ln_f, sigma_ln_f):
#     return bilby.prior.Gaussian(mu=mu_ln_f, sigma=sigma_ln_f).prob(val=dataset['kernel:terms[0]:log_f'])


def hyper_prior_log_f(dataset, min_ln_f, delta_ln_f):
    return bilby.prior.Uniform(minimum=min_ln_f, maximum=min_ln_f+delta_ln_f).prob(dataset['kernel:terms[0]:log_f'])
#
#
# def hyper_prior_log_c(dataset, mu_ln_c, sigma_ln_c):
#     return bilby.prior.Gaussian(mu=mu_ln_c, sigma=sigma_ln_c).prob(val=dataset['kernel:log_c'])
#
#
# def hyper_prior_log_a(dataset, mu_ln_a, sigma_ln_a):
#     return bilby.prior.Gaussian(mu=mu_ln_a, sigma=sigma_ln_a).prob(val=dataset['kernel:log_a'])

# def hyper_prior_log_f(dataset, min_ln_f, max_ln_f):
#     if min_ln_f > max_ln_f:
#         return 0
#     return bilby.prior.Uniform(minimum=min_ln_f, maximum=max_ln_f).prob(dataset['kernel:terms[0]:log_f'])


# def hyper_prior_log_c_qpo(dataset, mu_ln_c_qpo_1, sigma_ln_c_qpo_1, mu_ln_c_qpo_2, sigma_ln_c_qpo_2, eta):
#     p1 = eta * bilby.prior.Gaussian(mu=mu_ln_c_qpo_1, sigma=sigma_ln_c_qpo_1).prob(val=dataset['kernel:terms[0]:log_c'])
#     p2 = (1 - eta) * bilby.prior.Gaussian(mu=mu_ln_c_qpo_2, sigma=sigma_ln_c_qpo_2).prob(val=dataset['kernel:terms[0]:log_c'])
#     return p1 + p2
# def hyper_prior_log_c_qpo(dataset, mu_ln_c_qpo, sigma_ln_c_qpo):
#     return bilby.prior.Gaussian(mu=mu_ln_c_qpo, sigma=sigma_ln_c_qpo).prob(val=dataset['kernel:terms[0]:log_c'])


def hyper_prior_log_c_qpo(dataset, min_ln_c_qpo, delta_ln_c_qpo):
    return bilby.prior.Uniform(minimum=min_ln_c_qpo, maximum=min_ln_c_qpo+delta_ln_c_qpo).prob(dataset['kernel:terms[0]:log_c'])


def hyper_prior_log_a_qpo(dataset, min_ln_a_qpo, delta_ln_a_qpo):
    return bilby.prior.Uniform(minimum=min_ln_a_qpo, maximum=min_ln_a_qpo+delta_ln_a_qpo).prob(val=dataset['kernel:terms[0]:log_a'])

# def hyper_prior_log_a_qpo(dataset, mu_ln_a_qpo, sigma_ln_a_qpo):
#     return bilby.prior.Gaussian(mu=mu_ln_a_qpo, sigma=sigma_ln_a_qpo).prob(val=dataset['kernel:terms[0]:log_a'])

def hyper_prior_log_c_red_noise(dataset, min_ln_c_red_noise, delta_ln_c_red_noise):
    return bilby.prior.Uniform(minimum=min_ln_c_red_noise, maximum=min_ln_c_red_noise+delta_ln_c_red_noise).prob(val=dataset['kernel:terms[1]:log_c'])


def hyper_prior_log_a_red_noise(dataset, min_ln_a_red_noise, delta_ln_a_red_noise):
    return bilby.prior.Uniform(minimum=min_ln_a_red_noise, maximum=min_ln_a_red_noise+delta_ln_a_red_noise).prob(val=dataset['kernel:terms[1]:log_a'])

# def hyper_prior_log_c_red_noise(dataset, mu_ln_c_red_noise, sigma_ln_c_red_noise):
#     return bilby.prior.Gaussian(mu=mu_ln_c_red_noise, sigma=sigma_ln_c_red_noise).prob(
#         val=dataset['kernel:terms[1]:log_c'])
#
#
# def hyper_prior_log_a_red_noise(dataset, mu_ln_a_red_noise, sigma_ln_a_red_noise):
#     return bilby.prior.Gaussian(mu=mu_ln_a_red_noise, sigma=sigma_ln_a_red_noise).prob(
#         val=dataset['kernel:terms[1]:log_a'])


# hp = bilby.hyper.model.Model(model_functions=[hyper_prior_log_c, hyper_prior_log_a])
hp = bilby.hyper.model.Model(model_functions=[hyper_prior_log_a_qpo, hyper_prior_log_c_qpo,
                                              hyper_prior_log_a_red_noise, hyper_prior_log_c_red_noise,
                                              hyper_prior_log_f])
# hp = bilby.hyper.model.Model(model_functions=[hyper_prior_log_f])

min_log_a = -5
max_log_a = 15
min_log_c = -6
sampling_frequency = 256
run_priors = bilby.core.prior.PriorDict()

run_priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='terms[0]:log_a')
run_priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(sampling_frequency * 16),
                                                               name='terms[0]:log_c')
run_priors['kernel:terms[0]:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum), maximum=np.log(band_maximum),
                                                               name='terms[0]:log_f')
run_priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='terms[1]:log_a')
run_priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(sampling_frequency * 16),
                                                               name='terms[1]:log_c')
val = run_priors.prob(run_priors.sample())


def run_prior(dataset):
    return val


samples = [result.posterior for result in results]
# evidences = [result.log_evidence for result in results]
evidences = [result.log_bayes_factor - result_white_noise.log_bayes_factor for result, result_white_noise in zip(results, white_noise_results)]
print(evidences)
hp_likelihood = HyperparameterLikelihood(
    posteriors=samples, hyper_prior=hp,
    sampling_prior=run_prior, log_evidences=evidences, max_samples=500)

# hp_priors = dict(
#     mu_ln_a=Uniform(minimum=-5, maximum=15, name='mu_ln_a', latex_label='$\mu_{ln a}$'),
#     sigma_ln_a=Uniform(minimum=0, maximum=10, name='sigma_ln_a', latex_label='$\sigma_{ln_a}$'),
#     mu_ln_c=Uniform(minimum=-5, maximum=7, name='mu_ln_c', latex_label='$\mu_{ln c}$'),
#     sigma_ln_c=Uniform(minimum=0, maximum=10, name='sigma_ln_c', latex_label='$\sigma_{ln c}$'),
#     mu_ln_f=Uniform(np.log(5), np.log(64), 'mu_ln_f', '$\mu_{ln f}$'),
#     sigma_ln_f=Uniform(0, 2.5, 'sigma_ln_f', '$\sigma_{ln f}$'))
# hp_priors = dict(
#     mu_ln_a=Uniform(minimum=-5, maximum=15, name='mu_ln_a', latex_label='$\mu_{ln a}$'),
#     sigma_ln_a=Uniform(minimum=0, maximum=10, name='sigma_ln_a', latex_label='$\sigma_{ln_a}$'),
#     mu_ln_c=Uniform(minimum=-5, maximum=7, name='mu_ln_c', latex_label='$\mu_{ln c}$'),
#     sigma_ln_c=Uniform(minimum=0, maximum=10, name='sigma_ln_c', latex_label='$\sigma_{ln c}$'))
# hp_priors = dict(
    # mu_ln_a_qpo=Uniform(minimum=-5, maximum=15, name='mu_ln_a_qpo', latex_label='$\mu_{ln a}$ qpo'),
    # sigma_ln_a_qpo=Uniform(minimum=0, maximum=10, name='sigma_ln_a_qpo', latex_label='$\sigma_{ln_a}$ qpo'),
    # min_ln_c_qpo=Uniform(-5, 3, 'min_ln_f', '$\ln c_{min}$ qpo'),
    # max_ln_c_qpo=Uniform(-5, 3, 'max_ln_f', '$\ln c_{max}$ qpo'),
    # mu_ln_c_qpo_1=Uniform(minimum=-5, maximum=3, name='mu_ln_c_qpo_1', latex_label='$\mu_{ln c}$ qpo 1'),
    # sigma_ln_c_qpo_1=Uniform(minimum=0, maximum=10, name='sigma_ln_c_qpo_1', latex_label='$\sigma_{ln c}$ qpo 1'),
    # mu_ln_c_qpo=Uniform(minimum=-5, maximum=10, name='mu_ln_c_qpo_1', latex_label='$\mu_{ln c}$ qpo'),
    # sigma_ln_c_qpo=Uniform(minimum=0, maximum=10, name='sigma_ln_c_qpo_1', latex_label='$\sigma_{ln c}$ qpo'),
    # mu_ln_c_qpo_2=Uniform(minimum=3, maximum=7, name='mu_ln_c_qpo_2', latex_label='$\mu_{ln c}$ qpo 2'),
    # sigma_ln_c_qpo_2=Uniform(minimum=0, maximum=10, name='sigma_ln_c_qpo_2', latex_label='$\sigma_{ln c}$ qpo 2'),
    # eta=Uniform(minimum=0, maximum=1, name='eta_c_qpo', latex_label='$\eta_{c}$ qpo'),
    # mu_ln_a_red_noise=Uniform(minimum=-5, maximum=15, name='mu_ln_a red noise', latex_label='$\mu_{ln a}$ red noise'),
    # sigma_ln_a_red_noise=Uniform(minimum=0, maximum=10, name='sigma_ln_a red noise',
    #                              latex_label='$\sigma_{ln_a}$ red noise'),
    # mu_ln_c_red_noise=Uniform(minimum=-5, maximum=7, name='mu_ln_c red noise', latex_label='$\mu_{ln c}$ red noise'),
    # sigma_ln_c_red_noise=Uniform(minimum=0, maximum=10, name='sigma_ln_c red noise',
    #                              latex_label='$\sigma_{ln c}$ red noise'),
    # mu_ln_f=Uniform(np.log(5), np.log(64), 'mu_ln_f', '$\mu_{ln f}$'),
    # sigma_ln_f=Uniform(0, 2.5, 'sigma_ln_f', '$\sigma_{ln f}$'))
max_log_c = np.log(sampling_frequency * 16)

hp_priors = dict(
    min_ln_f_qpo=bilby.core.prior.Uniform(np.log(band_minimum), np.log(band_maximum), '$\ln f_{min}$ qpo'),
    delta_ln_f_qpo=bilby.core.prior.Uniform(0, np.log(band_maximum) - np.log(band_minimum), '$\ln f_{max}$ qpo'),
    min_ln_c_qpo=bilby.core.prior.Uniform(min_log_c, max_log_c, '$\ln c_{min}$ qpo'),
    delta_ln_c_qpo=bilby.core.prior.Uniform(0, max_log_c - min_log_c, '$\ln c_{max}$ qpo'),
    min_ln_a_qpo=bilby.core.prior.Uniform(min_log_a, max_log_a, '$\ln a_{min}$ qpo'),
    delta_ln_a_qpo=bilby.core.prior.Uniform(0, max_log_a - min_log_a, '$\ln a_{max}$ qpo'),
    min_ln_c_red_noise=bilby.core.prior.Uniform(min_log_c, max_log_c, '$\ln c_{min}$ red_noise'),
    delta_ln_c_red_noise=bilby.core.prior.Uniform(0, max_log_c - min_log_c, '$\ln c_{max}$ red_noise'),
    min_ln_a_red_noise=bilby.core.prior.Uniform(min_log_a, max_log_a, '$\ln a_{min}$ red_noise'),
    delta_ln_a_red_noise=bilby.core.prior.Uniform(0, max_log_a - min_log_a, '$\ln a_{max}$ red_noise')
)
# min_ln_f=Uniform(np.log(5), np.log(64), 'min_ln_f', '$\ln f_{min}$'),
# max_ln_f=Uniform(np.log(5), np.log(64), 'max_ln_f', '$\ln f_{max}$'))

# And run sampler
result = bilby.run_sampler(
    likelihood=hp_likelihood, priors=hp_priors, sampler='dynesty', nlive=1000,
    use_ratio=False, outdir=outdir, label=label, resume=False, sample='rslice')
result.plot_corner()
