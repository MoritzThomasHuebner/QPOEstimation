import json
import numpy as np

import bilby
from bilby.core.prior import Prior, ConditionalPriorDict, ConditionalBeta, Uniform, Beta


class MinimumPrior(ConditionalBeta):
    def __init__(self, order: int, minimum: float = 0, maximum: float = 1, name: str = None,
                 minimum_spacing: float = 0, latex_label: str = None, unit: str = None, boundary: str = None,
                 reference_name: str = None) -> None:
        """ A Conditional Beta prior that implements the conditional probabilities of Uniform order statistics

        Parameters
        ----------
        order:
            The order number of the parameter.
        minimum:
            The minimum of the prior range.
        maximum:
            The maximum of the prior range.
        name:
            The name of the prior.
        minimum_spacing:
            The minimal time-difference between two flares.
        latex_label:
            The latex label for the corner plot.
        unit:
            The unit for the corner plot.
        boundary:
            The boundary behaviour for the sampler. Must be from ['reflective', 'periodic', None].
        reference_name:
            The reference parameter name, which would be the prior with order of one less than the current.
        """

        super().__init__(
            alpha=1, beta=order, minimum=minimum, maximum=maximum,
            name=name, latex_label=latex_label, unit=unit,
            boundary=boundary, condition_func=self.minimum_condition
        )
        self.reference_params["order"] = order
        self.reference_params["minimum_spacing"] = minimum_spacing
        self.order = order

        if reference_name is None:
            self.reference_name = self.name[:-1] + str(int(self.name[-1]) - 1)
        else:
            self.reference_name = reference_name
        self._required_variables = [self.reference_name]
        self.minimum_spacing = minimum_spacing
        self.__class__.__name__ = "MinimumPrior"
        self.__class__.__qualname__ = "MinimumPrior"

    def minimum_condition(self, reference_params, **kwargs):  # noqa
        return dict(minimum=kwargs[self.reference_name] + self.minimum_spacing)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)

    def to_json(self):
        self.reset_to_reference_parameters()
        return json.dumps(self, cls=bilby.utils.BilbyJsonEncoder)
