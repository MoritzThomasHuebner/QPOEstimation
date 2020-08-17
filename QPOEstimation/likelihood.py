import bilby
import numpy as np
from bilby.core.likelihood import Likelihood
from celerite import terms

from QPOEstimation.model.psd import red_noise, white_noise, broken_power_law_noise, lorentzian


class ParameterAccessor(object):
    def __init__(self, parameter_name):
        self.parameter_name = parameter_name

    def __get__(self, instance, owner):
        return instance.parameters[self.parameter_name]

    def __set__(self, instance, value):
        instance.parameters[self.parameter_name] = value


class WhittleLikelihood(Likelihood):

    VALID_NOISE_MODELS = ['red_noise', 'broken_power_law']
    alpha = ParameterAccessor('alpha')
    alpha_1 = ParameterAccessor('alpha_1')
    alpha_2 = ParameterAccessor('alpha_2')
    beta = ParameterAccessor('beta')
    sigma = ParameterAccessor('sigma')
    rho = ParameterAccessor('rho')
    delta = ParameterAccessor('delta')
    amplitude = ParameterAccessor('amplitude')
    width = ParameterAccessor('width')
    central_frequency = ParameterAccessor('central_frequency')
    offset = ParameterAccessor('offset')

    def __init__(self, frequencies, periodogram, frequency_mask, noise_model='red_noise'):
        super(WhittleLikelihood, self).__init__(
            parameters=dict(alpha=0, alpha_1=0, alpha_2=0, beta=0, sigma=0, delta=0, rho=0,
                            amplitude=0, width=0, central_frequency=127, offset=0))
        self.frequencies = frequencies
        self._periodogram = periodogram
        self.frequency_mask = frequency_mask
        self.noise_model = noise_model

    @property
    def frequencies(self):
        return self._frequencies[self.frequency_mask]

    @frequencies.setter
    def frequencies(self, frequencies):
        self._frequencies = frequencies

    @property
    def model(self):
        return self.lorentzian

    @property
    def periodogram(self):
        return self._periodogram[self.frequency_mask]

    def log_likelihood(self):
        psd = self.psd + self.model
        return -np.sum(np.log(psd) + self.periodogram / psd)

    @property
    def lorentzian(self):
        return lorentzian(self.frequencies, self.amplitude, self.central_frequency, self.width, self.offset)

    @property
    def noise_model(self):
        return self._noise_model

    @noise_model.setter
    def noise_model(self, noise_model):
        if noise_model in self.VALID_NOISE_MODELS:
            self._noise_model = noise_model
        else:
            raise ValueError('Unknown noise model')

    @property
    def psd(self):
        if self.noise_model == 'red_noise':
            return red_noise(frequencies=self.frequencies, alpha=self.alpha, beta=self.beta) \
                   + white_noise(frequencies=self.frequencies, sigma=self.sigma)
        elif self.noise_model == 'broken_power_law':
            return broken_power_law_noise(frequencies=self.frequencies, alpha_1=self.alpha_1,
                                          alpha_2=self.alpha_2, beta=self.beta, delta=self.delta, rho=self.rho) \
                   + white_noise(frequencies=self.frequencies, sigma=self.sigma)


class CeleriteLikelihood(bilby.likelihood.Likelihood):

    def __init__(self, gp, y):
        parameters = gp.get_parameter_dict()
        self.gp = gp
        self.y = y
        super().__init__(parameters)

    def log_likelihood(self):
        # self.gp.set_parameter_vector(vector=self.parameters)
        for name, value in self.parameters.items():
            self.gp.set_parameter(name=name, value=value)
        return self.gp.log_likelihood(self.y)


class QPOTerm(terms.Term):
    parameter_names = ("log_a", "log_b", "log_c", "log_P")

    def get_real_coefficients(self, params):
        log_a, log_b, log_c, log_P = params
        b = np.exp(log_b)
        return (
            np.exp(log_a) * (1.0 + b) / (2.0 + b), np.exp(log_c),
        )

    def get_complex_coefficients(self, params):
        log_a, log_b, log_c, log_P = params
        b = np.exp(log_b)
        return (
            np.exp(log_a) / (2.0 + b), 0.0,
            np.exp(log_c), 2*np.pi*np.exp(-log_P),
        )