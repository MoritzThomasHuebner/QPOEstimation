import numpy as np
from bilby.core.likelihood import Likelihood


def red_noise(frequencies, alpha, beta):
    return beta * frequencies ** (-alpha)


def white_noise(frequencies, sigma):
    return sigma * np.ones(len(frequencies))


def broken_power_law_noise(frequencies, alpha_1, alpha_2, beta, delta, rho):
    return beta*frequencies**(-alpha_1) * (1 + (frequencies/delta)**((alpha_2-alpha_1)/rho))**(-rho)


def lorentzian(frequencies, amplitude, central_frequency, width):
    return amplitude * (width ** 2 / ((frequencies - central_frequency) ** 2 + width ** 2)) / np.pi / width


class QPLikelihood(Likelihood):

    VALID_NOISE_MODELS = ['red_noise', 'broken_power_law']

    def __init__(self, frequencies, amplitudes, frequency_mask, noise_model='red_noise'):
        super(QPLikelihood, self).__init__(
            parameters=dict(alpha=0, alpha_1=0, alpha_2=0, beta=0, sigma=0, delta=0, rho=0,
                            amplitude=0, width=1, central_frequency=127))
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.frequency_mask = frequency_mask
        self.noise_model = noise_model

    @property
    def psd(self):
        if self.noise_model == 'red_noise':
            return red_noise(frequencies=self.frequencies[self.frequency_mask], alpha=self.alpha, beta=self.beta) \
                   + white_noise(frequencies=self.frequencies[self.frequency_mask], sigma=self.sigma)
        elif self.noise_model == 'broken_power_law':
            return broken_power_law_noise(frequencies=self.frequencies[self.frequency_mask], alpha_1=self.alpha_1, alpha_2=self.alpha_2, beta=self.beta, delta=self.delta, rho=self.rho) \
                   + white_noise(frequencies=self.frequencies[self.frequency_mask], sigma=self.sigma)

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
    def lorentzian(self):
        return lorentzian(self.frequencies[self.frequency_mask], self.amplitude, self.central_frequency, self.width)

    @property
    def alpha(self):
        return self.parameters['alpha']

    @property
    def alpha_1(self):
        return self.parameters['alpha_1']

    @property
    def alpha_2(self):
        return self.parameters['alpha_2']

    @property
    def beta(self):
        return self.parameters['beta']

    @property
    def delta(self):
        return self.parameters['delta']

    @property
    def rho(self):
        return self.parameters['rho']

    @property
    def sigma(self):
        return self.parameters['sigma']

    @property
    def amplitude(self):
        return self.parameters['amplitude']

    @property
    def width(self):
        return self.parameters['width']

    @property
    def central_frequency(self):
        return self.parameters['central_frequency']

    @property
    def model(self):
        return self.lorentzian

    @property
    def residual(self):
        return self.amplitudes[self.frequency_mask] - self.model

    def log_likelihood(self):
        psd = self.psd
        return np.sum(-self.residual ** 2 / psd / 2 - np.log(2 * np.pi * psd) / 2)


class ParameterAccessor(object):
    pass
