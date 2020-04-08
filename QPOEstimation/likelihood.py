import numpy as np
from bilby.core.likelihood import Likelihood


def red_noise(frequencies, alpha, beta):
    return beta * frequencies ** (-alpha)


def white_noise(frequencies, sigma):
    return sigma * np.ones(len(frequencies))


def lorentzian(frequencies, amplitude, central_frequency, width):
    return amplitude * (width ** 2 / ((frequencies - central_frequency) ** 2 + width ** 2)) / np.pi / width


class QPLikelihood(Likelihood):
    def __init__(self, frequencies, amplitudes, frequency_mask):
        super(QPLikelihood, self).__init__(
            parameters=dict(alpha=0, beta=0, sigma=0, amplitude=0, width=0, central_frequency=0))
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.frequency_mask = frequency_mask

    @property
    def psd(self):
        return red_noise(self.frequencies[self.frequency_mask], self.alpha, self.beta) \
               + white_noise(self.frequencies[self.frequency_mask], self.sigma)

    @property
    def lorentzian(self):
        return lorentzian(self.frequencies[self.frequency_mask], self.amplitude, self.central_frequency, self.width)

    @property
    def alpha(self):
        return self.parameters['alpha']

    @property
    def beta(self):
        return self.parameters['beta']

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