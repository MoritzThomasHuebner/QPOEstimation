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


class ParameterAccessor(object):
    def __init__(self, parameter_name):
        self.parameter_name = parameter_name

    def __get__(self, instance, owner):
        return instance.parameters[self.parameter_name]

    def __set__(self, instance, value):
        instance.parameters[self.parameter_name] = value


class QPLikelihood(Likelihood):

    VALID_NOISE_MODELS = ['red_noise', 'broken_power_law']
    alpha = ParameterAccessor('alpha')
    alpha_1 = ParameterAccessor('alpha_1')
    alpha_2 = ParameterAccessor('alpha_2')
    log_beta = ParameterAccessor('log_beta')
    log_sigma = ParameterAccessor('log_sigma')
    rho = ParameterAccessor('rho')
    delta = ParameterAccessor('delta')
    log_amplitude = ParameterAccessor('log_amplitude')
    log_width = ParameterAccessor('log_width')
    central_frequency = ParameterAccessor('central_frequency')

    def __init__(self, frequencies, amplitudes, frequency_mask, noise_model='red_noise'):
        super(QPLikelihood, self).__init__(
            parameters=dict(alpha=0, alpha_1=0, alpha_2=0, log_beta=0, log_sigma=0, delta=0, rho=0,
                            log_amplitude=0, log_width=0, central_frequency=127))
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.frequency_mask = frequency_mask
        self.noise_model = noise_model

    @property
    def frequencies(self):
        return self._frequencies[self.frequency_mask]

    @frequencies.setter
    def frequencies(self, frequencies):
        self._frequencies = frequencies

    @property
    def psd(self):
        if self.noise_model == 'red_noise':
            return red_noise(frequencies=self.frequencies, alpha=self.alpha, beta=self.beta) \
                   + white_noise(frequencies=self.frequencies, sigma=self.sigma)
        elif self.noise_model == 'broken_power_law':
            return broken_power_law_noise(frequencies=self.frequencies, alpha_1=self.alpha_1,
                                          alpha_2=self.alpha_2, beta=self.beta, delta=self.delta, rho=self.rho) \
                   + white_noise(frequencies=self.frequencies, sigma=self.sigma)

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
        return lorentzian(self.frequencies, self.amplitude, self.central_frequency, self.width)

    @property
    def beta(self):
        return np.exp(self.log_beta)

    @property
    def sigma(self):
        return np.exp(self.log_sigma)

    @property
    def amplitude(self):
        return np.exp(self.log_amplitude)

    @property
    def width(self):
        return np.exp(self.log_width)

    @property
    def model(self):
        return self.lorentzian

    @property
    def residual(self):
        return self.amplitudes[self.frequency_mask] - self.model

    def log_likelihood(self):
        psd = self.psd
        return np.sum(-self.residual ** 2 / psd / 2 - np.log(2 * np.pi * psd) / 2)


class PeriodogramQPOLikelihood(QPLikelihood):

    def __init__(self, frequencies, periodogram, frequency_mask, noise_model='red_noise'):
        super(PeriodogramQPOLikelihood, self).__init__(frequencies=frequencies, frequency_mask=frequency_mask,
                                                       noise_model=noise_model, amplitudes=None)
        del self.amplitudes
        self.frequencies = frequencies
        self._periodogram = periodogram
        self.frequency_mask = frequency_mask
        self.noise_model = noise_model

    @property
    def periodogram(self):
        return self._periodogram[self.frequency_mask]

    def log_likelihood(self):
        psd = self.psd + self.model
        return -np.sum(np.log(psd) + self.periodogram / psd)


