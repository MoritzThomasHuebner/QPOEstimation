import numpy as np
import matplotlib.pyplot as plt
import bilby
from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import Uniform
from bilby.core.sampler import run_sampler
from bilby.core.result import make_pp_plot
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.hyper.model import Model

recovery_mode = 'qpo'
# outdir = 'testing_hyper_pe_qpo'
outdir = f'testing_hyper_pe_{recovery_mode}'
label = 'no_ln_f'

results = bilby.result.ResultList([bilby.result.read_in_result(
    f'sliding_window_5_64Hz_smoothed_residual/period_{i}/{recovery_mode}/results/20_gaussian_process_result.json')
    for i in range(5, 26)])


# def hyper_prior_log_f(dataset, mu_ln_f, sigma_ln_f):
#     return bilby.prior.Gaussian(mu=mu_ln_f, sigma=sigma_ln_f).prob(val=dataset['kernel:log_f'])

def hyper_prior_log_f(dataset, min_ln_f, max_ln_f):
    if min_ln_f > max_ln_f:
        return 0
    return bilby.prior.Uniform(minimum=min_ln_f, maximum=max_ln_f).prob(dataset['kernel:log_f'])


def hyper_prior_log_c(dataset, mu_ln_c, sigma_ln_c):
    return bilby.prior.Gaussian(mu=mu_ln_c, sigma=sigma_ln_c).prob(val=dataset['kernel:log_c'])


def hyper_prior_log_a(dataset, mu_ln_a, sigma_ln_a):
    return bilby.prior.Gaussian(mu=mu_ln_a, sigma=sigma_ln_a).prob(val=dataset['kernel:log_a'])


hp = bilby.hyper.model.Model(model_functions=[hyper_prior_log_c, hyper_prior_log_a])
# hp = bilby.hyper.model.Model(model_functions=[hyper_prior_log_f])


def run_prior(dataset):
    return 1 / 11.54517744448 / 20# / 2.54944517093


samples = [result.posterior for result in results]
# evidences = [result.log_evidence for result in results]
evidences = [result.log_bayes_factor for result in results]
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
hp_priors = dict(
    mu_ln_a=Uniform(minimum=-5, maximum=15, name='mu_ln_a', latex_label='$\mu_{ln a}$'),
    sigma_ln_a=Uniform(minimum=0, maximum=10, name='sigma_ln_a', latex_label='$\sigma_{ln_a}$'),
    mu_ln_c=Uniform(minimum=-5, maximum=7, name='mu_ln_c', latex_label='$\mu_{ln c}$'),
    sigma_ln_c=Uniform(minimum=0, maximum=10, name='sigma_ln_c', latex_label='$\sigma_{ln c}$'))
# hp_priors = dict(
#     mu_ln_a=Uniform(minimum=-5, maximum=15, name='mu_ln_a', latex_label='$\mu_{ln a}$'),
#     sigma_ln_a=Uniform(minimum=0, maximum=10, name='sigma_ln_a', latex_label='$\sigma_{ln_a}$'),
#     mu_ln_c=Uniform(minimum=-5, maximum=7, name='mu_ln_c', latex_label='$\mu_{ln c}$'),
#     sigma_ln_c=Uniform(minimum=0, maximum=10, name='sigma_ln_c', latex_label='$\sigma_{ln c}$'),
#     min_ln_f=Uniform(np.log(5), np.log(64), 'min_ln_f', '$\ln f_{min}$'),
#     max_ln_f=Uniform(np.log(5), np.log(64), 'max_ln_f', '$\ln f_{max}$'))

# And run sampler
result = run_sampler(
    likelihood=hp_likelihood, priors=hp_priors, sampler='dynesty', nlive=300,
    use_ratio=False, outdir=outdir, label=label, resume=True)
result.plot_corner()
