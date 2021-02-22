import numpy as np

from celerite.modeling import Model

import QPOEstimation
from QPOEstimation.model.series import exponential_background, burst_envelope, polynomial, gaussian, log_normal, \
    lorentzian, stabilised_burst_envelope, stabilised_exponential
import bilby
from bilby.core.prior import Uniform, ConditionalPriorDict, Beta


def function_to_celerite_mean_model(func):
    class CeleriteMeanModel(Model):
        parameter_names = tuple(bilby.core.utils.infer_args_from_function_except_n_args(func=func, n=1))

        def get_value(self, t):
            params = {name: getattr(self, name) for name in self.parameter_names}
            return func(t, **params)

        def compute_gradient(self, *args, **kwargs):
            pass

    return CeleriteMeanModel


PolynomialMeanModel = function_to_celerite_mean_model(polynomial)
ExponentialMeanModel = function_to_celerite_mean_model(exponential_background)
GaussianMeanModel = function_to_celerite_mean_model(gaussian)
LogNormalMeanModel = function_to_celerite_mean_model(log_normal)
LorentzianMeanModel = function_to_celerite_mean_model(lorentzian)
FastRiseExponentialDecayMeanModel = function_to_celerite_mean_model(burst_envelope)
StabilisedFastRiseExponentialDecayMeanModel = function_to_celerite_mean_model(stabilised_burst_envelope)
ExponentialStabilisedMeanModel = function_to_celerite_mean_model(stabilised_exponential)


def get_n_component_mean_model(model, n_models=1, defaults=None):
    base_names = bilby.core.utils.infer_args_from_function_except_n_args(func=model, n=1)
    names = []
    for i in range(n_models):
        for base in base_names:
            names.extend([f"{base}_{i}"])
    names = tuple(names)
    if defaults is None:
        defaults = dict()
        for name in names:
            defaults[name] = 0.1

    class MultipleMeanModel(Model):
        parameter_names = names

        def get_value(self, t):
            res = np.zeros(len(t))
            for i in range(n_models):
                res += model(t, **{f"{base}": getattr(self, f"{base}_{i}") for base in base_names})
            return res

        def compute_gradient(self, *args, **kwargs):
            pass

    return MultipleMeanModel(**defaults)


def get_n_component_fred_model(n_freds=1):
    return get_n_component_mean_model(model=burst_envelope, n_models=n_freds)


def get_n_component_stabilised_fred_model(n_freds=1):
    return get_n_component_mean_model(model=stabilised_burst_envelope, n_models=n_freds)


def get_fred_priors(n_freds=1, t_min=0, t_max=2000, minimum_spacing=0):
    priors = ConditionalPriorDict()

    if n_freds == 1:
        priors[f'mean:amplitude_0'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e12, name='A')
        priors[f'mean:sigma_0'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=10000, name='sigma')
        priors[f'mean:skewness_0'] = bilby.core.prior.LogUniform(minimum=np.exp(-20), maximum=np.exp(20),
                                                                    name=f'skewness_0')
        priors[f'mean:t_max_0'] = Uniform(t_min, t_max, name="t_max")
        return priors
    for ii in range(n_freds):
        if ii == 0:
            priors[f"mean:t_max_{ii}"] = Beta(minimum=t_min, maximum=t_max, alpha=1, beta=n_freds, name=f"mean:t_max_{ii}")
        else:
            priors[f"mean:t_max_{ii}"] = QPOEstimation.prior.minimum.MinimumPrior(
                order=n_freds-ii, minimum_spacing=minimum_spacing, minimum=t_min, maximum=t_max, name=f"mean:t_max_{ii}")
        priors[f'mean:amplitude_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=1e12, name=f'A_{ii}')
        priors[f'mean:sigma_{ii}'] = bilby.core.prior.LogUniform(minimum=1e-3, maximum=10000, name=f'sigma_{ii}')
        priors[f'mean:skewness_{ii}'] = bilby.core.prior.LogUniform(minimum=np.exp(-20), maximum=np.exp(20), name=f'log skewness_{ii}')
        priors[f"mean:t_max_{ii}"].__class__.__name__ = "MinimumPrior"
    return priors
