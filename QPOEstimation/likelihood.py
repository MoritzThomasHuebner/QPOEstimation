import numpy as np
from bilby.core.likelihood import Likelihood, function_to_celerite_mean_model, function_to_george_mean_model
import celerite
import george
from celerite import terms
from typing import Union

from QPOEstimation.model import mean_model_dict
from QPOEstimation.model.psd import red_noise, white_noise, broken_power_law_noise, lorentzian
from QPOEstimation.model.celerite import get_n_component_mean_model
from QPOEstimation.model.mean import polynomial

from bilby.core.likelihood import CeleriteLikelihood, GeorgeLikelihood


class ParameterAccessor(object):
    def __init__(self, parameter_name: str) -> None:
        """ Handy accessor for the likelihood parameter dict so we can call them as if they are attributes.

        Parameters
        ----------
        parameter_name:
            The name of the parameter.
        """
        self.parameter_name = parameter_name

    def __get__(self, instance, owner):
        return instance.parameters[self.parameter_name]

    def __set__(self, instance, value):
        instance.parameters[self.parameter_name] = value


class WhittleLikelihood(Likelihood):
    VALID_NOISE_MODELS = ["red_noise", "broken_power_law", "pure_qpo", "white_noise"]
    alpha = ParameterAccessor("alpha")
    alpha_1 = ParameterAccessor("alpha_1")
    alpha_2 = ParameterAccessor("alpha_2")
    log_beta = ParameterAccessor("log_beta")
    log_sigma = ParameterAccessor("log_sigma")
    rho = ParameterAccessor("rho")
    log_delta = ParameterAccessor("log_delta")
    log_amplitude = ParameterAccessor("log_amplitude")
    log_width = ParameterAccessor("log_width")
    log_frequency = ParameterAccessor("log_frequency")

    def __init__(
            self, frequencies: np.ndarray, periodogram: np.ndarray, frequency_mask: np.ndarray,
            noise_model: str = "red_noise") -> None:
        """ A Whittle likelihood class for use with `bilby`.

        Parameters
        ----------
        frequencies:
            The periodogram frequencies.
        periodogram:
            The periodogram powers.
        frequency_mask:
            A mask we can apply if we want to mask out certain frequencies/powers.
            Provide as indices which to retain.
        noise_model:
            The noise model. Should be 'red_noise' or 'broken_power_law'.
        """
        super(WhittleLikelihood, self).__init__(
            parameters=dict(alpha=0, alpha_1=0, alpha_2=0, log_beta=0, log_sigma=0, log_delta=0, rho=0,
                            log_amplitude=0, log_width=1, log_frequency=127))
        self.frequencies = frequencies
        self._periodogram = periodogram
        self.frequency_mask = frequency_mask
        self.noise_model = noise_model

    @property
    def beta(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_beta)

    @property
    def sigma(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_sigma)

    @property
    def delta(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_delta)

    @property
    def amplitude(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_amplitude)

    @property
    def width(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_width)

    @property
    def frequency(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_frequency)

    @property
    def frequencies(self) -> np.ndarray:
        return self._frequencies[self.frequency_mask]

    @frequencies.setter
    def frequencies(self, frequencies: np.ndarray) -> None:
        self._frequencies = frequencies

    @property
    def model(self) -> np.ndarray:
        return self.lorentzian

    @property
    def periodogram(self) -> np.ndarray:
        return self._periodogram[self.frequency_mask]

    def log_likelihood(self) -> float:
        """ Calculates the log-likelihood.

        Returns
        -------
        The log-likelihood.
        """
        psd = self.psd + self.model
        return -np.sum(np.log(psd) + self.periodogram / psd)

    @property
    def lorentzian(self) -> np.ndarray:
        return lorentzian(self.frequencies, self.amplitude, self.frequency, self.width)

    @property
    def noise_model(self) -> str:
        return self._noise_model

    @noise_model.setter
    def noise_model(self, noise_model: str) -> None:
        if noise_model in self.VALID_NOISE_MODELS:
            self._noise_model = noise_model
        elif noise_model == "qpo_plus_red_noise":
            self._noise_model = "red_noise"
        else:
            raise ValueError(f"Unknown noise model {noise_model}")

    @property
    def psd(self) -> np.ndarray:
        if self.noise_model == "red_noise":
            return red_noise(frequencies=self.frequencies, alpha=self.alpha, beta=self.beta) \
                   + white_noise(frequencies=self.frequencies, sigma=self.sigma)
        elif self.noise_model in ["pure_qpo", "white_noise"]:
            return white_noise(frequencies=self.frequencies, sigma=self.sigma)
        elif self.noise_model == "broken_power_law":
            return broken_power_law_noise(frequencies=self.frequencies, alpha_1=self.alpha_1,
                                          alpha_2=self.alpha_2, beta=self.beta, delta=self.delta, rho=self.rho) \
                   + white_noise(frequencies=self.frequencies, sigma=self.sigma)


class WindowedCeleriteLikelihood(CeleriteLikelihood):

    def __init__(
            self, mean_model: celerite.modeling.Model, kernel: celerite.terms.Term, t: np.ndarray,
            y: np.ndarray, yerr: np.ndarray) -> None:
        """
        `celerite` to `bilby` likelihood interface for GP that has defined start and end time within series.
        The likelihood adds two parameters 'window_minimum' and 'window_maximum'. Inside this window we apply the GP.
        Outside we only assume white noise.

        Parameters
        ----------
        mean_model:
            The celerite mean model.
        kernel:
            The celerite kernel.
        t:
            The time coordinates.
        y:
            The y-values.
        yerr:
            The y-error-values.
        """
        super(WindowedCeleriteLikelihood, self).__init__(
            kernel=kernel, mean_model=mean_model, t=t, y=y, yerr=yerr)
        self.parameters["window_minimum"] = t[0]
        self.parameters["window_maximum"] = t[-1]

        self._white_noise_kernel = celerite.terms.JitterTerm(log_sigma=-20)
        self.white_noise_gp = celerite.GP(kernel=self._white_noise_kernel, mean=self.mean_model)
        self.white_noise_gp.compute(self.gp._t, self.yerr) # noqa
        self.white_noise_log_likelihood = self.white_noise_gp.log_likelihood(y=y)

    def log_likelihood(self) -> Union[float, np.ndarray]:
        """

        Returns
        -------
        The log-likelihood.
        """
        if self._check_valid_indices_distribution():
            return -np.inf

        self._setup_gps()

        log_l = self.gp.log_likelihood(self.y[self.windowed_indices]) + \
            self.white_noise_gp.log_likelihood(self.y[self.edge_indices])
        return np.nan_to_num(log_l, nan=-np.inf)

    def _check_valid_indices_distribution(self) -> bool:
        return len(self.windowed_indices) == 0 or len(self.edge_indices) == 0

    def _setup_gps(self) -> None:
        self.gp.compute(self.t[self.windowed_indices], self.yerr[self.windowed_indices])
        self.white_noise_gp.compute(self.t[self.edge_indices], self.yerr[self.edge_indices] + self.jitter)
        self._set_parameters_to_gps()

    def _set_parameters_to_gps(self) -> None:
        for name, value in self.parameters.items():
            if "window" in name:
                continue
            if "mean" in name:
                self.white_noise_gp.set_parameter(name=name, value=value)
            self.gp.set_parameter(name=name, value=value)

    @property
    def jitter(self) -> float:
        for k in self.parameters.keys():
            if k.endswith("log_sigma"):
                return np.exp(self.parameters[k])
        return 0

    @property
    def edge_indices(self) -> np.ndarray:
        return np.where(np.logical_or(self.window_minimum > self.t, self.t > self.window_maximum))[0]

    @property
    def windowed_indices(self) -> np.ndarray:
        return np.where(np.logical_and(self.window_minimum < self.t, self.t < self.window_maximum))[0]

    @property
    def window_minimum(self) -> float:
        return self.parameters["window_minimum"]

    @property
    def window_maximum(self) -> float:
        return self.parameters["window_maximum"]

    def noise_log_likelihood(self) -> float:
        """ log-likelihood assuming everything is white noise.

        Returns
        -------
        The noise log-likelihood.
        """
        return self.white_noise_log_likelihood


class QPOTerm(terms.Term):
    """ Kernel with equal amplitude and damping time exponential and cosine component.
    Proposed in the `celerite` paper, but we don't use it. """
    parameter_names = ("log_a", "log_b", "log_c", "log_f")

    def get_real_coefficients(self, params: tuple) -> tuple:
        log_a, log_b, log_c, log_f = params
        b = np.exp(log_b)
        return (
            np.exp(log_a) * (1.0 + b) / (2.0 + b), np.exp(log_c),
        )

    def get_complex_coefficients(self, params: tuple) -> tuple:
        log_a, log_b, log_c, log_f = params
        b = np.exp(log_b)
        return (
            np.exp(log_a) / (2.0 + b), 0.0,
            np.exp(log_c), 2 * np.pi * np.exp(log_f),
        )

    def compute_gradient(self, *args, **kwargs):
        pass


class ExponentialTerm(terms.Term):
    """ Exponential kernel that we use as our red noise model. """
    parameter_names = ("log_a", "log_c")

    def get_real_coefficients(self, params: tuple) -> tuple:
        log_a, log_c = params
        b = np.exp(10)
        return (
            np.exp(log_a) * (1.0 + b) / (2.0 + b), np.exp(log_c),
        )

    def get_complex_coefficients(self, params: tuple) -> tuple:
        log_a, log_c = params
        b = np.exp(10)
        return (
            np.exp(log_a) / (2.0 + b), 0.0,
            np.exp(log_c), 50,
        )

    def compute_gradient(self, *args, **kwargs):
        pass


class PureQPOTerm(terms.Term):
    """ Exponential kernel that we use as our red noise model. """
    parameter_names = ("log_a", "log_c", "log_f")

    def get_real_coefficients(self, params: tuple) -> tuple:
        log_a, log_c, log_f = params
        return 0, np.exp(log_c),

    def get_complex_coefficients(self, params: tuple) -> tuple:
        log_a, log_c, log_f = params
        a = np.exp(log_a)
        c = np.exp(log_c)
        f = np.exp(log_f)
        return a, 0.0, c, 2 * np.pi * f,

    def compute_gradient(self, *args, **kwargs):
        pass


def get_kernel(kernel_type: str, jitter_term: bool = False) -> Union[celerite.terms.Term, george.kernels.Kernel]:
    """ Catch all kernel getter.

    Parameters
    ----------
    kernel_type: The name of the kernel. Must be from `QPOEstimation.MODES`.
    jitter_term: Whether to add a `JitterTerm`, i.e. an additional white noise term.

    Returns
    -------
    The kernel.
    """
    if kernel_type == "white_noise":
        return celerite.terms.JitterTerm(log_sigma=-20)
    elif kernel_type == "qpo":
        res = QPOTerm(log_a=0.1, log_b=-10, log_c=-0.01, log_f=3)
    elif kernel_type == "pure_qpo":
        res = PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3)
    elif kernel_type == "red_noise":
        res = ExponentialTerm(log_a=0.1, log_c=-0.01)
    elif kernel_type == "qpo_plus_red_noise":
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
    elif kernel_type == "sho":
        res = celerite.terms.SHOTerm(log_S0=1, log_Q=0, log_omega0=0)
    elif kernel_type == "double_sho":
        res = celerite.terms.SHOTerm(log_S0=1, log_Q=0, log_omega0=0) + \
              celerite.terms.SHOTerm(log_S0=1, log_Q=0, log_omega0=0)
    elif kernel_type == "matern32":
        res = george.kernels.Matern32Kernel(metric=1.0) * george.kernels.ConstantKernel(log_constant=0)
    elif kernel_type == "matern52":
        res = george.kernels.Matern52Kernel(metric=1.0) * george.kernels.ConstantKernel(log_constant=0)
    elif kernel_type == "exp_sine2":
        res = george.kernels.ExpSine2Kernel(gamma=1.0, log_period=10.0) * george.kernels.ConstantKernel(log_constant=0)
    elif kernel_type == "rational_quadratic":
        res = george.kernels.RationalQuadraticKernel(log_alpha=0.0, metric=1.0)
    elif kernel_type == "exp_squared":
        res = george.kernels.ExpSquaredKernel(metric=1.0) * george.kernels.ConstantKernel(log_constant=0)
    elif kernel_type == "exp_sine2_rn":
        res = george.kernels.ExpSine2Kernel(gamma=1.0, log_period=10.0) * george.kernels.ConstantKernel(log_constant=0)\
              + george.kernels.ExpKernel(metric=1.0) * george.kernels.ConstantKernel(log_constant=0)
    else:
        raise ValueError("Recovery mode not defined")

    if jitter_term:
        res += celerite.terms.JitterTerm(log_sigma=-20)

    return res


def get_mean_model(
        model_type: Union[float, str], n_components: int = 1, y: Union[float, np.ndarray] = None, offset: bool = False,
        likelihood_model: str = "celerite") -> Union[celerite.modeling.Model, george.modeling.Model]: # noqa
    """ Creates a mean model instance for use in the likelihood.

    Parameters
    ----------
    model_type:
        The model type as a string. Must be from `QPOEstimation.model.mean`.
    n_components:
        The number of flare shapes to use.
    y:
        The y-coordinates of the data. Only relevant if we use a constant mean as a mean model.
    offset:
        If we are using a constant offset component.
    likelihood_model:
        The likelihood model we use. Must be from ['celerite', 'celerite_windowed', 'george'].

    Returns
    -------
    The mean model.
    """
    if model_type == "polynomial":
        if likelihood_model in ["celerite", "celerite_windowed"]:
            return function_to_celerite_mean_model(polynomial)(a0=0, a1=0, a2=0, a3=0, a4=0)
        elif likelihood_model == "george":
            return function_to_george_mean_model(polynomial)(a0=0, a1=0, a2=0, a3=0, a4=0)
    elif model_type == "mean":
        return np.mean(y)
    elif isinstance(model_type, (int, float)) or model_type.isnumeric():
        return float(model_type)
    elif model_type in mean_model_dict:
        return get_n_component_mean_model(mean_model_dict[model_type], n_models=n_components, offset=offset,
                                          likelihood_model=likelihood_model)
    else:
        raise ValueError


def get_gp_likelihood(
        mean_model: Union[celerite.modeling.Model, george.modeling.Model], # noqa
        kernel: Union[celerite.terms.Term, george.kernels.Kernel], times: np.ndarray, y: np.ndarray, yerr:
        np.ndarray, likelihood_model: str = "celerite")\
        -> Union[CeleriteLikelihood, GeorgeLikelihood, WindowedCeleriteLikelihood]:
    """ Creates the correct likelihood instance for the inference process.
    
    Parameters
    ----------
    mean_model:
        The mean model we use.
    kernel:
        The kernel function.
    times:
        The time coordinates.
    y:
        The y-values.
    yerr:
        The y-error values.
    likelihood_model:
        The likelihood model. Must be from `QPOEstimation.LIKELIHOOD_MODELS`.

    Returns
    -------
    The instance of the likelihood class.
    """
    return LIKELIHOOD_MODEL_DICT[likelihood_model](mean_model=mean_model, kernel=kernel, t=times, y=y, yerr=yerr)


LIKELIHOOD_MODEL_DICT = dict(
    celerite=CeleriteLikelihood, celerite_windowed=WindowedCeleriteLikelihood, george=GeorgeLikelihood)
