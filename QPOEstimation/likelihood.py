import bilby
import numpy as np
from bilby.core.likelihood import Likelihood
from celerite import terms
from scipy.special import gammaln, gamma


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


class GrothLikelihood(WhittleLikelihood):

    def __init__(self, frequencies, periodogram, noise_model='red_noise'):
        super(GrothLikelihood, self).__init__(frequencies=frequencies, periodogram=periodogram,
                                              frequency_mask=[True]*len(frequencies), noise_model=noise_model)

    # def log_likelihood(self):
    #     log_l = -np.sum(self.psd + self.model)
    #     groth_term_prev = 0
    #     sum_diff = 1e-20
    #     m = 0
    #     groth_sum = 0.
    #     while(True):
    #         groth_sum_factor = self.psd**m * self.model**m
    #         groth_term = groth_sum_factor/(gamma(m + 1))**2
    #         if np.abs(np.sum(groth_term - groth_term_prev)) < sum_diff:
    #             break
    #         groth_term_prev = groth_term
    #         groth_sum += groth_term
    #         m += 1
    #     log_l += np.sum(np.log(groth_sum))
    #     return log_l

    @property
    def psd(self):
        if self.noise_model == 'red_noise':
            return red_noise(frequencies=self.frequencies, alpha=self.alpha, beta=self.beta)
        elif self.noise_model == 'broken_power_law':
            return broken_power_law_noise(frequencies=self.frequencies, alpha_1=self.alpha_1,
                                          alpha_2=self.alpha_2, beta=self.beta, delta=self.delta, rho=self.rho)

    def log_likelihood(self):
        log_l = -np.sum(self.psd + self.model)

        groth_sum = 0.
        for m in range(20):
            groth_sum_factor = self.psd**m * self.model**m
            groth_term = groth_sum_factor/(gamma(m + 1))**2
            groth_sum += groth_term
            # print(log_l)
        log_l += np.sum(np.log(groth_sum))
        return log_l


class CeleriteLikelihood(bilby.likelihood.Likelihood):

    def __init__(self, gp, y, conversion_func=None):
        parameters = gp.get_parameter_dict()
        if conversion_func is None:
            self.conversion_func = lambda x: x
        else:
            self.conversion_func = conversion_func
        self.gp = gp
        self.y = y
        super().__init__(parameters)

    def log_likelihood(self):
        celerite_params = self.conversion_func(self.parameters)
        # self.gp.set_parameter_vector(vector=self.parameters)
        for name, value in celerite_params.items():
            self.gp.set_parameter(name=name, value=value)
        try:
            return self.gp.log_likelihood(self.y)
        except Exception:
            return -np.inf


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
            np.exp(log_c), 2 * np.pi * np.exp(-log_P),
        )


class ZeroedQPOTerm(terms.Term):
    parameter_names = ("log_a", "log_c", "log_f")

    def get_real_coefficients(self, params):
        log_a, log_c, log_f = params
        return 0, np.exp(log_c),

    def get_complex_coefficients(self, params):
        log_a, log_c, log_f = params
        a = np.exp(log_a)
        c = np.exp(log_c)
        f = np.exp(log_f)
        return a, 0.0, c, 2 * np.pi * f,


class PoissonLikelihoodWithBackground(bilby.core.likelihood.PoissonLikelihood):

    def __init__(self, x, y, func, background):
        super(PoissonLikelihoodWithBackground, self).__init__(x=x, y=y, func=func)
        self.background = background

    def noise_log_likelihood(self):
        return self._eval_log_likelihood(self.background)

    def log_likelihood(self):
        rate = self.func(self.x, **self.model_parameters) + self.background
        rate[np.where(rate <= 0)] = 1e-30
        return self._eval_log_likelihood(rate)

    def _eval_log_likelihood(self, rate):
        return np.sum(-rate + self.y * np.log(rate) - gammaln(self.y + 1))


class AssociationLikelihood(bilby.core.likelihood.Likelihood):

    def __init__(self, posteriors):
        super().__init__(parameters=dict(log_f_0=0))
        self.posteriors = posteriors

    def log_likelihood(self):
        return np.sum(np.log(self.p_associated_any()))

    def p_associated_any(self):
        ps = []
        for prob in self.posteriors:
            ps.append(1 - self.p_unassociated(prob))
        return ps

    def p_unassociated(self, prob):
        p = 1
        ll = 2
        frequency = 0
        while frequency < prob.maximum:
            frequency = self.frequency_at_mode(ll=ll)
            if prob.minimum <= frequency:
                p_associated = prob.prob(frequency) * 1 / (ll + 1)  # 1/l correction factor?
                if ll == 1:
                    p_associated = 0
                p *= 1 - p_associated
            ll += 1
        return p

    def frequency_at_mode(self, ll):
        return np.exp(self.parameters['log_f_0']) * np.sqrt(ll * (ll + 1))
