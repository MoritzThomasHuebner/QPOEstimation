import numpy as np
from typing import Union

from george.modeling import Model as GeorgeModel
from celerite.modeling import Model as CeleriteModel

import bilby

LIKELIHOOD_MODEL_DICT = dict(george=GeorgeModel, celerite=CeleriteModel, celerite_windowed=CeleriteModel)


def _get_parameter_names(base_names: list, n_models: int, offset: bool) -> tuple:
    """ Takes a list of parameter names and modifies them to account for a multiple component model.

    Parameters
    ----------
    base_names:
        The parameter names of the basis function.
    n_models:
        Number of flare shapes we want to compound.
    offset:
        Whether we want to add an offset parameter

    Returns
    -------
    The parameter names of the `celerite` or `george` model.
    """
    names = []
    for i in range(n_models):
        for base in base_names:
            names.extend([f"{base}_{i}"])
    if offset:
        names.extend(["offset"])
    return tuple(names)


def get_n_component_mean_model(
        model: callable, n_models: int = 1, defaults: dict = None, offset: bool = False,
        likelihood_model: str = "celerite") -> Union:
    """ Takes a function and turns it into an n component `celerite` or `george` mean model.

    Parameters
    ----------
    model:
        The model with the x-coordinate as the first function argument.
    n_models:
        Number of flare shapes we want to compound.
    defaults:
        Default values of the function parameters.
    offset:
        Whether we want to include a constant offset in the model.
    likelihood_model:
        'celerite' or 'george'

    Returns
    -------
    The `celerite` or `george` mean model
    """
    base_names = bilby.core.utils.infer_parameters_from_function(func=model)
    names = _get_parameter_names(base_names, n_models, offset)
    defaults = defaults or {name: 0.1 for name in names}

    m = LIKELIHOOD_MODEL_DICT[likelihood_model]

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
