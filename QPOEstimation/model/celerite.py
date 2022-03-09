import numpy as np

import bilby


def _get_model_class(likelihood_model="gaussian_process"):
    if likelihood_model == "george":
        from george.modeling import Model as GeorgeModel
        return GeorgeModel
    else:
        from celerite.modeling import Model as CeleriteModel
        return CeleriteModel


def _get_names(base_names, n_models, offset):
    names = []
    for i in range(n_models):
        for base in base_names:
            names.extend([f"{base}_{i}"])
    if offset:
        names.extend(["offset"])
    names = tuple(names)
    return names


def get_n_component_mean_model(model, n_models=1, defaults=None, offset=False, likelihood_model="gaussian_process"):
    base_names = bilby.core.utils.infer_parameters_from_function(func=model)
    names = _get_names(base_names, n_models, offset)
    defaults = defaults or {name: 0.1 for name in names}

    m = _get_model_class(likelihood_model=likelihood_model)

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
