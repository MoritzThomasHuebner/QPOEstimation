import numpy as np

from celerite.modeling import Model as CeleriteModel
from george.modeling import Model as GeorgeModel

from QPOEstimation.model.mean import fred, polynomial, gaussian, log_normal, \
    lorentzian
import bilby


def function_to_celerite_mean_model(func):
    return function_to_model(func, CeleriteModel)


def function_to_george_mean_model(func):
    return function_to_model(func, GeorgeModel)


def function_to_model(func, cls):
    class MeanModel(cls):
        parameter_names = tuple(bilby.core.utils.infer_args_from_function_except_n_args(func=func, n=1))

        def get_value(self, t):
            params = {name: getattr(self, name) for name in self.parameter_names}
            return func(t, **params)

        def compute_gradient(self, *args, **kwargs):
            pass

    return MeanModel


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


# def get_n_component_piecewise(n_components=6, likelihood_model='gaussian_process', mode='linear'):
#     names = [f'beta_{i}' for i in range(n_components)]
#     if mode == 'linear':
#         names.extend([f'k_{i}' for i in range(2, n_components)])
#     elif mode == 'cubic':
#         names.extend([f'k_{i}' for i in range(4, n_components)])
#
#     names = tuple(names)
#     defaults = dict()
#     for name in names:
#         defaults[name] = 0.0
#
#     if likelihood_model == 'george_likelihood':
#         m = GeorgeModel
#     else:
#         m = CeleriteModel
#
#     def _linear(times, beta, k):
#         res = beta * (times - k)
#         res[np.where(times < k)] = 0
#         return res
#
#     def _cubic(times, beta, k):
#         res = beta * (times - k) ** 3
#         res[np.where(times < k)] = 0
#         return res
#
#     class PiecewiseLinearMeanModel(m):
#         parameter_names = names
#
#         def get_value(self, t):
#             duration = t[-1] - t[0]
#             times = 2 * t / duration - 1
#             betas = np.array([getattr(self, f"beta_{i}") for i in range(n_components)])
#             ks = 2 * np.array([getattr(self, f"k_{i}", 0) for i in range(n_components)]) / duration - 1
#
#             return betas[0] + np.sum([_linear(times, betas[i], ks[i]) for i in range(1, len(betas))], axis=0)
#
#         def compute_gradient(self, *args, **kwargs):
#             pass
#
#     class PiecewiseCubicMeanModel(m):
#         parameter_names = names
#
#         def get_value(self, t):
#             deltas = np.array([delta_0, delta_1, delta_2])
#             ks = np.array([0, k_1, k_2])
#             alphas = [alpha_0]
#             betas = [beta_0]
#             gammas = [gamma_0]
#             for i in range(1, len(deltas)):
#                 diff = (ks[i] - ks[i - 1])
#                 alphas.append(
#                     alphas[i - 1] + betas[i - 1] * diff + gammas[i - 1] * diff ** 2 + deltas[i - 1] * diff ** 3)
#                 betas.append(betas[i - 1] + 2 * gammas[i - 1] * diff + 3 * deltas[i - 1] * diff ** 2)
#                 gammas.append(2 * gammas[i - 1] + 6 * deltas[i - 1] * diff)
#             res = np.zeros(len(times))
#             indices = np.append(np.searchsorted(times, ks), len(times))
#             indices_list = [np.arange(indices[i], indices[i + 1]) for i in range(len(betas))]
#             for i in range(len(betas)):
#                 adjusted_times = times[indices_list[i]] - ks[i]
#                 res[indices_list[i]] = alphas[i] + betas[i] * adjusted_times + gammas[i] * adjusted_times ** 2 + \
#                                        deltas[i] * adjusted_times ** 3
#             return res
#
#
#
#             duration = t[-1] - t[0]
#             times = 2 * t / duration - 1
#             betas = np.array([getattr(self, f"beta_{i}") for i in range(n_components)])
#             ks = 2 * np.array([getattr(self, f"k_{i}", 0) for i in range(n_components)]) / duration - 1
#             return betas[0] + betas[1] * times + betas[2] * times ** 2 + np.sum(
#                 [_cubic(times, betas[i], ks[i]) for i in range(3, len(betas))], axis=0)
#
#         def compute_gradient(self, *args, **kwargs):
#             pass
#
#     if mode == 'linear':
#         return PiecewiseLinearMeanModel(**defaults)
#     else:
#         return PiecewiseCubicMeanModel(**defaults)

def get_n_component_piecewise(n_components=6, likelihood_model='gaussian_process', mode='linear'):
    names = [f'delta_{i}' for i in range(n_components)]
    names.extend([f'k_{i}' for i in range(1, n_components)])
    names.extend(['alpha_0', 'beta_0', 'gamma_0'])

    names = tuple(names)
    defaults = dict()
    for name in names:
        defaults[name] = 0.0

    if likelihood_model == 'george_likelihood':
        m = GeorgeModel
    else:
        m = CeleriteModel


    class PiecewiseCubicMeanModel(m):
        parameter_names = names

        def get_value(self, t):
            duration = t[-1] - t[0]
            times = 2 * t / duration - 1

            deltas = np.array([getattr(self, f"delta_{i}") for i in range(n_components)])
            ks = 2 * np.array([getattr(self, f"k_{i}", 0) for i in range(n_components)]) / duration - 1
            alphas = [getattr(self, f"alpha_0")]
            betas = [getattr(self, f"beta_0")]
            gammas = [getattr(self, f"gamma_0")]
            for i in range(1, len(deltas)):
                diff = (ks[i] - ks[i - 1])
                alphas.append(
                    alphas[i - 1] + betas[i - 1] * diff + gammas[i - 1] * diff ** 2 + deltas[i - 1] * diff ** 3)
                betas.append(betas[i - 1] + 2 * gammas[i - 1] * diff + 3 * deltas[i - 1] * diff ** 2)
                gammas.append(2 * gammas[i - 1] + 6 * deltas[i - 1] * diff)
            res = np.zeros(len(times))
            indices = np.append(np.searchsorted(times, ks), len(times))
            indices_list = [np.arange(indices[i], indices[i + 1]) for i in range(len(betas))]
            for i in range(len(betas)):
                adjusted_times = times[indices_list[i]] - ks[i]
                res[indices_list[i]] = alphas[i] + betas[i] * adjusted_times + gammas[i] * adjusted_times ** 2 + \
                                       deltas[i] * adjusted_times ** 3
            return res

        def compute_gradient(self, *args, **kwargs):
            pass

    return PiecewiseCubicMeanModel(**defaults)

def get_n_component_fred_model(n_freds=1):
    return get_n_component_mean_model(model=fred, n_models=n_freds)


def power_qpo(a, c, f):
    return a * np.sqrt((c**2 + 2 * np.pi**2 * f**2)/(c * (c**2 + 4 * np.pi**2 * f**2)))


def power_red_noise(a, c):
    return a / c**0.5
