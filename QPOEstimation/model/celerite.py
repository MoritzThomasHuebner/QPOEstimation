import numpy as np

from celerite.modeling import Model

from QPOEstimation.model.series import exponential_background, fred, polynomial, gaussian, log_normal, \
    lorentzian
import bilby


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
FREDMeanModel = function_to_celerite_mean_model(fred)


def get_n_component_mean_model(model, n_models=1, defaults=None, offset=False):
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

    class MultipleMeanModel(Model):
        parameter_names = names

        def get_value(self, t):
            res = np.zeros(len(t))
            for i in range(n_models):
                res += model(t, **{f"{base}": getattr(self, f"{base}_{i}") for base in base_names})
            if offset:
                res += getattr(self, "offset")
            return res

        def compute_gradient(self, *args, **kwargs):
            pass

    return MultipleMeanModel(**defaults)


def get_n_component_fred_model(n_freds=1):
    return get_n_component_mean_model(model=fred, n_models=n_freds)


