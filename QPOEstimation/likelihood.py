import bilby
import numpy as np
from bilby.core.likelihood import Likelihood
import celerite
from celerite import terms
import george
from scipy.special import gamma


from QPOEstimation.model import mean_model_dict
from QPOEstimation.model.psd import red_noise, white_noise, broken_power_law_noise, lorentzian
from QPOEstimation.model.celerite import PolynomialMeanModel, get_n_component_mean_model, \
    get_n_component_piecewise


def get_celerite_likelihood(mean_model, kernel, fit_mean, times, y, yerr, likelihood_model='gaussian_process'):
    return LIKELIHOOD_MODELS[likelihood_model](mean_model=mean_model, kernel=kernel,
                                               fit_mean=fit_mean, t=times, y=y, yerr=yerr)


class ParameterAccessor(object):
    def __init__(self, parameter_name):
        self.parameter_name = parameter_name

    def __get__(self, instance, owner):
        return instance.parameters[self.parameter_name]

    def __set__(self, instance, value):
        instance.parameters[self.parameter_name] = value


class WhittleLikelihood(Likelihood):
    VALID_NOISE_MODELS = ['red_noise', 'broken_power_law', 'pure_qpo', 'white_noise']
    alpha = ParameterAccessor('alpha')
    alpha_1 = ParameterAccessor('alpha_1')
    alpha_2 = ParameterAccessor('alpha_2')
    log_beta = ParameterAccessor('log_beta')
    log_sigma = ParameterAccessor('log_sigma')
    rho = ParameterAccessor('rho')
    log_delta = ParameterAccessor('log_delta')
    log_amplitude = ParameterAccessor('log_amplitude')
    log_width = ParameterAccessor('log_width')
    log_frequency = ParameterAccessor('log_frequency')

    def __init__(self, frequencies, periodogram, frequency_mask, noise_model='red_noise'):
        super(WhittleLikelihood, self).__init__(
            parameters=dict(alpha=0, alpha_1=0, alpha_2=0, log_beta=0, log_sigma=0, log_delta=0, rho=0,
                            log_amplitude=0, log_width=1, log_frequency=127))
        self.frequencies = frequencies
        self._periodogram = periodogram
        self.frequency_mask = frequency_mask
        self.noise_model = noise_model

    @property
    def beta(self):
        return np.exp(self.log_beta)

    @property
    def sigma(self):
        return np.exp(self.log_sigma)

    @property
    def delta(self):
        return np.exp(self.log_delta)

    @property
    def amplitude(self):
        return np.exp(self.log_amplitude)

    @property
    def width(self):
        return np.exp(self.log_width)

    @property
    def frequency(self):
        return np.exp(self.log_frequency)

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
        log_l = -np.sum(np.log(psd) + self.periodogram / psd)
        # if np.isnan(log_l):
        #     print(log_l)
        return log_l

    @property
    def lorentzian(self):
        return lorentzian(self.frequencies, self.amplitude, self.frequency, self.width)

    @property
    def noise_model(self):
        return self._noise_model

    @noise_model.setter
    def noise_model(self, noise_model):
        if noise_model in self.VALID_NOISE_MODELS:
            self._noise_model = noise_model
        elif noise_model == 'general_qpo':
            self._noise_model = 'red_noise'
        else:
            raise ValueError(f'Unknown noise model {noise_model}')

    @property
    def psd(self):
        if self.noise_model == 'red_noise':
            return red_noise(frequencies=self.frequencies, alpha=self.alpha, beta=self.beta) \
                   + white_noise(frequencies=self.frequencies, sigma=self.sigma)
        elif self.noise_model in ['pure_qpo', 'white_noise']:
            return white_noise(frequencies=self.frequencies, sigma=self.sigma)
        elif self.noise_model == 'broken_power_law':
            return broken_power_law_noise(frequencies=self.frequencies, alpha_1=self.alpha_1,
                                          alpha_2=self.alpha_2, beta=self.beta, delta=self.delta, rho=self.rho) \
                   + white_noise(frequencies=self.frequencies, sigma=self.sigma)


class GrothLikelihood(WhittleLikelihood):

    def __init__(self, frequencies, periodogram, noise_model='red_noise'):
        super(GrothLikelihood, self).__init__(frequencies=frequencies, periodogram=periodogram,
                                              frequency_mask=[True]*len(frequencies), noise_model=noise_model)

    def log_likelihood(self):
        log_l = -np.sum(self.psd + self.model + self.periodogram)
        for i, freq in enumerate(self.frequencies):
            groth_sum = 0
            for m in range(20):
                groth_sum_factor = (self.psd[i] + self.model[i])**m * self.periodogram[i]**m
                groth_term = groth_sum_factor / (gamma(m + 1)) ** 2
                groth_sum += groth_term
                if groth_term < 1e-20:
                    break
            log_l += np.log(groth_sum)
        return log_l

    @property
    def psd(self):
        if self.noise_model == 'red_noise':
            return red_noise(frequencies=self.frequencies, alpha=self.alpha, beta=self.beta)
        elif self.noise_model == 'broken_power_law':
            return broken_power_law_noise(frequencies=self.frequencies, alpha_1=self.alpha_1,
                                          alpha_2=self.alpha_2, beta=self.beta, delta=self.delta, rho=self.rho)


class CeleriteLikelihood(bilby.likelihood.Likelihood):

    def __init__(self, kernel, mean_model, fit_mean, t, y, yerr, conversion_func=None):
        """ Celerite to bilby likelihood interface """

        self.kernel = kernel
        self.mean_model = mean_model
        self.fit_mean = fit_mean
        self.t = np.array(t)
        self.y = np.array(y)
        self.y_err = np.array(yerr)

        if conversion_func is None:
            self.conversion_func = lambda x: x
        else:
            self.conversion_func = conversion_func
        self.gp = celerite.GP(kernel=kernel, mean=mean_model, fit_mean=fit_mean)
        self.gp.compute(t=t, yerr=yerr)

        self._white_noise_kernel = celerite.terms.JitterTerm(log_sigma=-20)
        self.white_noise_gp = celerite.GP(kernel=self._white_noise_kernel, mean=self.mean_model, fit_mean=self.fit_mean)
        self.white_noise_gp.compute(self.gp._t, self.y_err)
        self.white_noise_log_likelihood = self.white_noise_gp.log_likelihood(y=y)
        super().__init__(parameters=self.gp.get_parameter_dict())

    def log_likelihood(self):
        celerite_params = self.conversion_func(self.parameters)
        self.gp.set_parameter_vector(vector=np.array(list(celerite_params.values())))
        # for name, value in celerite_params.items():
        #     try:
                # self.gp.set_parameter(name=name, value=value)
            # except ValueError:
            #     raise ValueError(f"Parameter {name} not a valid parameter for the GP.")
        try:
            return self.gp.log_likelihood(self.y)
        except Exception:
            return -np.inf

    def noise_log_likelihood(self):
        return self.white_noise_log_likelihood


class GeorgeLikelihood(bilby.likelihood.Likelihood):

    def __init__(self, kernel, mean_model, fit_mean, t, y, yerr, conversion_func=None):
        """ Celerite to bilby likelihood interface """

        self.kernel = kernel
        self.mean_model = mean_model
        self.fit_mean = fit_mean
        self.t = np.array(t)
        self.y = np.array(y)
        self.y_err = np.array(yerr)

        if conversion_func is None:
            self.conversion_func = lambda x: x
        else:
            self.conversion_func = conversion_func

        self.gp = george.GP(kernel=kernel, mean=mean_model, fit_mean=fit_mean, fit_white_noise=False)
        self.gp.compute(x=t, yerr=yerr)
        super().__init__(parameters=self.gp.get_parameter_dict())

    def log_likelihood(self):
        celerite_params = self.conversion_func(self.parameters)
        for name, value in celerite_params.items():
            try:
                self.gp.set_parameter(name=name, value=value)
            except ValueError:
                raise ValueError(f"Parameter {name} not a valid parameter for the GP.")
        try:
            return self.gp.log_likelihood(self.y)
        except Exception:
            return -np.inf


class WindowedCeleriteLikelihood(CeleriteLikelihood):

    def __init__(self, mean_model, kernel, fit_mean, t, y, yerr, conversion_func=None):
        """ Celerite to bilby likelihood interface for GP that has defined start and end time within series. """
        super(WindowedCeleriteLikelihood, self).__init__(kernel=kernel, mean_model=mean_model, fit_mean=fit_mean, t=t,
                                                         y=y, yerr=yerr, conversion_func=conversion_func)
        self.parameters['window_minimum'] = t[0]
        self.parameters['window_maximum'] = t[-1]

    def log_likelihood(self):
        if len(self.windowed_indices) == 0 or len(self.edge_indices) == 0:
            return -np.inf

        self.gp.compute(self.t[self.windowed_indices], self.y_err[self.windowed_indices])
        self.white_noise_gp.compute(self.t[self.edge_indices], self.y_err[self.edge_indices])
        celerite_params = self.conversion_func(self.parameters)
        for name, value in celerite_params.items():
            if 'window' in name:
                continue
            if 'mean' in name:
                self.white_noise_gp.set_parameter(name=name, value=value)
            self.gp.set_parameter(name=name, value=value)

        log_l = self.gp.log_likelihood(self.y[self.windowed_indices]) + \
            self.white_noise_gp.log_likelihood(self.y[self.edge_indices])
        return np.nan_to_num(log_l, nan=-np.inf)

    @property
    def edge_indices(self):
        return np.where(np.logical_or(self.window_minimum > self.t, self.t > self.window_maximum))[0]

    @property
    def windowed_indices(self):
        return np.where(np.logical_and(self.window_minimum < self.t, self.t < self.window_maximum))[0]

    @property
    def window_minimum(self):
        return self.parameters['window_minimum']

    @property
    def window_maximum(self):
        # return self.parameters['window_minimum'] + self.parameters['window_size']
        return self.parameters['window_maximum']

    def noise_log_likelihood(self):
        return self.white_noise_log_likelihood


class DoubleCeleriteLikelihood(Likelihood):

    def __init__(self, mean_model, kernel_0, kernel_1, fit_mean, t, y, y_err,
                 joint_parameters=None, conversion_func=None):
        """ Celerite to bilby likelihood interface for GP that has defined start and end time within series. """
        self.kernel_0 = kernel_0
        self.kernel_1 = kernel_1
        self.mean_model = mean_model
        self.fit_mean = fit_mean
        self.t = t
        self.y = y
        self.y_err = y_err

        if conversion_func is None:
            self.conversion_func = lambda x: x
        else:
            self.conversion_func = conversion_func

        self.parameters = dict(transition_time=0)
        self.gp_0 = celerite.GP(kernel=kernel_0, mean=mean_model, fit_mean=fit_mean)
        self.gp_0.compute(t=self.t[self.before_transition_indices], yerr=self.y_err[self.before_transition_indices])
        self.gp_1 = celerite.GP(kernel=kernel_1, mean=mean_model, fit_mean=fit_mean)
        self.gp_1.compute(t=self.t[self.after_transition_indices], yerr=self.y_err[self.after_transition_indices])
        if joint_parameters is None:
            self.joint_parameters = []
        else:
            self.joint_parameters = joint_parameters
        for name, val in self.gp_0.get_parameter_dict().items():
            if 'mean' in name:
                self.joint_parameters.append(name)

        parameters_0 = self.gp_0.get_parameter_dict()
        parameters_1 = self.gp_1.get_parameter_dict()
        parameters = dict()
        for param in self.joint_parameters:
            parameters[param] = parameters_0[param]
            del parameters_0[param]
            del parameters_1[param]

        for param, val in parameters_0.items():
            parameters[f"{param}_0"] = val

        for param, val in parameters_1.items():
            parameters[f"{param}_1"] = val

        super().__init__(parameters=parameters)
        self.parameters['transition_time'] = t[-1] - t[0]

    @property
    def before_transition_indices(self):
        return np.where(self.t < self.parameters['transition_time'])[0]

    @property
    def after_transition_indices(self):
        return np.where(self.t >= self.parameters['transition_time'])[0]

    def log_likelihood(self):
        if len(self.before_transition_indices) == 0 or len(self.after_transition_indices) == 0:
            return -np.inf
        self.gp_0.compute(t=self.t[self.before_transition_indices], yerr=self.y_err[self.before_transition_indices])
        self.gp_1.compute(t=self.t[self.after_transition_indices], yerr=self.y_err[self.after_transition_indices])

        celerite_params = self.conversion_func(self.parameters)
        for name, value in celerite_params.items():
            if name in self.joint_parameters:
                self.gp_0.set_parameter(name=name, value=value)
                self.gp_1.set_parameter(name=name, value=value)
            elif name.endswith("_0"):
                self.gp_0.set_parameter(name=name.rstrip("_0"), value=value)
            elif name.endswith("_1"):
                self.gp_1.set_parameter(name=name.rstrip("_1"), value=value)

        log_l = self.gp_0.log_likelihood(self.y[self.before_transition_indices]) + \
            self.gp_1.log_likelihood(self.y[self.after_transition_indices])
        return np.nan_to_num(log_l, nan=-np.inf)


class QPOTerm(terms.Term):
    parameter_names = ("log_a", "log_b", "log_c", "log_f")

    def get_real_coefficients(self, params):
        log_a, log_b, log_c, log_f = params
        b = np.exp(log_b)
        return (
            np.exp(log_a) * (1.0 + b) / (2.0 + b), np.exp(log_c),
        )

    def get_complex_coefficients(self, params):
        log_a, log_b, log_c, log_f = params
        b = np.exp(log_b)
        return (
            np.exp(log_a) / (2.0 + b), 0.0,
            np.exp(log_c), 2 * np.pi * np.exp(log_f),
        )


class ExponentialTerm(terms.Term):
    parameter_names = ("log_a", "log_c")

    def get_real_coefficients(self, params):
        log_a, log_c = params
        b = np.exp(10)
        return (
            np.exp(log_a) * (1.0 + b) / (2.0 + b), np.exp(log_c),
        )

    def get_complex_coefficients(self, params):
        log_a, log_c = params
        b = np.exp(10)
        return (
            np.exp(log_a) / (2.0 + b), 0.0,
            np.exp(log_c), 50,
        )


class PureQPOTerm(terms.Term):
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


def get_kernel(kernel_type, jitter_term=False):
    if kernel_type == "white_noise":
        return celerite.terms.JitterTerm(log_sigma=-20)
    elif kernel_type == "qpo":
        res = QPOTerm(log_a=0.1, log_b=-10, log_c=-0.01, log_f=3)
    elif kernel_type == "pure_qpo":
        res = PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3)
    elif kernel_type == "red_noise":
        res = ExponentialTerm(log_a=0.1, log_c=-0.01)
    elif kernel_type == "general_qpo":
        res = PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + ExponentialTerm(log_a=0.1, log_c=-0.01)
    elif kernel_type == "double_red_noise":
        res = ExponentialTerm(log_a=0.1, log_c=-0.01) + ExponentialTerm(log_a=0.1, log_c=-0.01)
    elif kernel_type == "double_qpo":
        res = PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3)
    elif kernel_type == "fourier_series":
        res = PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + \
              PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + \
              PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + \
              ExponentialTerm(log_a=0.1, log_c=-0.01)

    elif kernel_type == 'sho':
        res = celerite.terms.SHOTerm(log_S0=1, log_Q=0, log_omega0=0)
    elif kernel_type == 'double_sho':
        res = celerite.terms.SHOTerm(log_S0=1, log_Q=0, log_omega0=0) + \
              celerite.terms.SHOTerm(log_S0=1, log_Q=0, log_omega0=0)
    elif kernel_type == 'matern32':
        res = george.kernels.Matern32Kernel(metric=1.0)
    else:
        raise ValueError('Recovery mode not defined')

    if jitter_term:
        res += celerite.terms.JitterTerm(log_sigma=-20)
    return res


def get_mean_model(model_type, n_components=1, y=None, offset=False, likelihood_model='gaussian_process'):
    if model_type == 'polynomial':
        return PolynomialMeanModel(a0=0, a1=0, a2=0, a3=0, a4=0), True
    elif model_type == 'mean':
        return np.mean(y), False
    elif model_type.isnumeric():
        return float(model_type), False
    elif model_type == 'piecewise_linear':
        return get_n_component_piecewise(n_components=n_components, likelihood_model=likelihood_model, mode="linear"), True
    elif model_type == 'piecewise_cubic':
        return get_n_component_piecewise(n_components=n_components, likelihood_model=likelihood_model, mode="cubic"), True
    elif model_type in mean_model_dict:
        return get_n_component_mean_model(mean_model_dict[model_type], n_models=n_components, offset=offset,
                                          likelihood_model=likelihood_model), True
    else:
        raise ValueError


LIKELIHOOD_MODELS = dict(gaussian_process=CeleriteLikelihood, gaussian_process_windowed=WindowedCeleriteLikelihood,
                         george_likelihood=GeorgeLikelihood)
