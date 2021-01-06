from celerite.modeling import Model

from QPOEstimation.model.series import exponential_background
from QPOEstimation.stabilisation import bar_lev


class PolynomialMeanModel(Model):
    """ Celerite compatible polynomial mean model """

    parameter_names = ("a0", "a1", "a2", "a3", "a4")

    def get_value(self, t):
        times = t
        return self.a0 + self.a1 * times + self.a2 * times**2 + self.a3 * times**3 + self.a4 * times**4

    def compute_gradient(self, *args, **kwargs):
        pass


class ExponentialMeanModel(Model):
    """ Celerite compatible exponential mean model """
    parameter_names = ("tau", "offset")

    def get_value(self, t):
        return exponential_background(times=t, tau=self.tau, offset=self.offset)

    def compute_gradient(self, *args, **kwargs):
        pass


class ExponentialStabilisedMeanModel(Model):
    """ Celerite compatible exponential mean model """
    parameter_names = ("tau", "offset")

    def get_value(self, t):
        return bar_lev(exponential_background(times=t, tau=self.tau, offset=self.offset))

    def compute_gradient(self, *args, **kwargs):
        pass
