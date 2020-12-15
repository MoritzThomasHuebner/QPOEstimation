from bilby.core.prior import Prior, ConditionalPriorDict, ConditionalBeta, Uniform, Beta
import numpy as np


class MinimumPrior(ConditionalBeta):
    """
    Adopted from kookaburra. Needs to be imported at some stage
    """
    def __init__(self, order, minimum=0, maximum=1, name=None,
                 minimum_spacing=0, latex_label=None, unit=None, boundary=None,
                 reference_name=None):
        super().__init__(
            alpha=1, beta=order, minimum=minimum, maximum=maximum,
            name=name, latex_label=latex_label, unit=unit,
            boundary=boundary, condition_func=self.minimum_condition
        )
        self.order = order
        if reference_name is None:
            self.reference_name = self.name[:-1] + str(int(self.name[-1]) - 1)
        else:
            self.reference_name = reference_name
        self._required_variables = [self.reference_name]
        self.minimum_spacing = minimum_spacing
        self.__class__.__name__ = 'MinimumPrior'

    def minimum_condition(self, reference_params, **kwargs):
        return dict(minimum=kwargs[self.reference_name] + self.minimum_spacing)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)


def get_frequency_mode_priors(n_freqs=1, f_min=5, f_max=32, minimum_spacing=0):
    priors = ConditionalPriorDict()
    keys = [f'log_f_{i}' for i in range(n_freqs)]
    if n_freqs == 1:
        priors[keys[0]] = Uniform(minimum=np.log(f_min), maximum=np.log(f_max))
        return priors
    for ii, key in enumerate(keys):
        if ii == 0:
            priors[key] = Beta(
                minimum=np.log(f_min),
                maximum=np.log(f_max),
                alpha=1,
                beta=n_freqs,
                name=key,
                latex_label=key
            )
        else:
            priors[key] = MinimumPrior(
                order=n_freqs - ii,
                minimum_spacing=minimum_spacing,
                minimum=np.log(f_min),
                maximum=np.log(f_max),
                name=key,
                latex_label=key
            )
        priors[key].__class__.__name__ = "MinimumPrior"
    return priors