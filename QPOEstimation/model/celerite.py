import numpy as np

from celerite.modeling import Model as CeleriteModel
try:
    from george.modeling import Model as GeorgeModel
except ImportError:
    pass
from QPOEstimation.model.mean import fred, polynomial, gaussian, log_normal, \
    lorentzian
import bilby
from bilby.core.likelihood import function_to_celerite_mean_model


PolynomialMeanModel = function_to_celerite_mean_model(polynomial)
GaussianMeanModel = function_to_celerite_mean_model(gaussian)
LogNormalMeanModel = function_to_celerite_mean_model(log_normal)
LorentzianMeanModel = function_to_celerite_mean_model(lorentzian)
FREDMeanModel = function_to_celerite_mean_model(fred)


def get_n_component_mean_model(model, n_models=1, defaults=None, offset=False, likelihood_model='gaussian_process'):
    base_names = bilby.core.utils.infer_args_from_function_except_n_args(func=model, n=1)
    names = []
    for i in range(n_models):
        for base in base_names:
            names.extend([f"{base}_{i}"])
    if offset:
        names.extend(['offset'])

    names = tuple(names)
    if defaults is None:
        defaults = dict()
        for name in names:
            defaults[name] = 0.1

    if likelihood_model == 'george_likelihood':
        m = GeorgeModel
    else:
        m = CeleriteModel

    class MultipleMeanModel(m):
        parameter_names = names

        def get_value(self, t):
            res = np.zeros(len(t))
            for j in range(n_models):
                res += model(t, **{f"{b}": getattr(self, f"{b}_{j}") for b in base_names})
            if offset:
                res += getattr(self, "offset")
            return res

        def compute_gradient(self, *args, **kwargs):
            pass

    return MultipleMeanModel(**defaults)


def get_n_component_fred_model(n_freds=1):
    return get_n_component_mean_model(model=fred, n_models=n_freds)


def power_qpo(a, c, f):
    return a * np.sqrt((c**2 + 2 * np.pi**2 * f**2)/(c * (c**2 + 4 * np.pi**2 * f**2)))


def power_red_noise(a, c):
    return a / c**0.5
