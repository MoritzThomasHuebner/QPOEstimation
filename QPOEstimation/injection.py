import json
from pathlib import Path
from typing import Union

import bilby
import numpy as np
import celerite
import matplotlib.pyplot as plt

from QPOEstimation.likelihood import get_kernel, get_mean_model


class InjectionCreator(object):

    def __init__(
            self, params: dict, injection_mode: str, sampling_frequency: float = 256, segment_length: float = 1.,
            times: np.ndarray = None, outdir: str = "injection_files", injection_id: Union[int, str] = 0,
            likelihood_model: str = "celerite", mean_model: Union[float, str] = None, n_components: int = 1,
            poisson_data: bool = False) -> None:
        """ Data generation class. Use the `create_injection` as the preferred interface.

        Parameters
        ----------
        params:
            The parameters for the injection data.
        injection_mode:
            The kernel to use in the injection. Must be from `QPOEstimation.MODES`.
        sampling_frequency:
            The sampling frequency. Optional, only relevant if `times` is None.
        segment_length:
            The segment length. Optional, only relevant if `times` is None.
        times:
            The times. `sampling_frequency` and `segment_length` are easier to use if we want a
            equally spaced data set. We can choose arbitrary spacings by passing `times`.
        outdir:
            The directory to write the files into. (Default_value = 'injection_files').
        injection_id:
            The ID number of this data set.
        likelihood_model:
            The likelihood to use. Must be from `QPOEstimation.LIKELIHOOD_MODELS`. (Default_value = 'celerite')
        mean_model:
            The mean model to use. Must be from one of the functions in `QPOEstimation.model.mean`, or 'mean'
            or a number for a constant mean model.
        n_components:
            The number of mean model flares to use. (Default_value = 1)
        poisson_data:
            Whether to create Poisson data.
            This assumes the underlying true rate and creates Poisson process data from it.
        """
        self.params = params
        self.injection_mode = injection_mode
        self.sampling_frequency = sampling_frequency
        self.segment_length = segment_length
        self.outdir = outdir
        self.injection_id = injection_id
        self.poisson_data = poisson_data
        self.likelihood_model = likelihood_model
        self.kernel = get_kernel(self.injection_mode)

        self.times = times
        self.mean_model = get_mean_model(model_type=mean_model, n_components=n_components, y=0)
        for key, value in params.items():
            if key.startswith("mean"):
                self.mean_model.__setattr__(key.replace("mean:", ""), value)
        self.gp = celerite.GP(kernel=self.kernel, mean=self.mean_model)
        self.gp.compute(self.windowed_times, self.windowed_yerr)
        self.update_params()
        self.y_realisation = self.gp.mean.get_value(self.times)
        self.y_realisation[self.windowed_indices] = self.gp.sample()
        self.y_realisation[self.outside_window_indices] += \
            np.random.normal(size=len(self.outside_window_indices)) * self.yerr[self.outside_window_indices]
        self.create_outdir()

    def create_outdir(self) -> None:
        """ Creates directory structure for the injection data. """
        Path(f"{self.outdir}/{self.injection_mode}/{self.likelihood_model}").mkdir(exist_ok=True, parents=True)

    @property
    def times(self) -> np.ndarray:
        return self._times

    @times.setter
    def times(self, times: np.ndarray) -> None:
        if times is None:
            self._times = np.linspace(0, self.segment_length, int(self.sampling_frequency * self.segment_length))
        else:
            self._times = times

    @property
    def windowed_indices(self) -> np.ndarray:
        if self.likelihood_model == "celerite":
            return np.where(np.logical_and(-np.inf < self.times, self.times < np.inf))[0]
        else:
            return np.where(np.logical_and(self.params["window_minimum"] < self.times,
                                           self.times < self.params["window_maximum"]))[0]

    @property
    def outside_window_indices(self) -> np.ndarray:
        if self.likelihood_model == "celerite_windowed":
            return np.where(np.logical_or(self.params["window_minimum"] >= self.times,
                                          self.times >= self.params["window_maximum"]))[0]
        else:
            return np.array([])

    @property
    def windowed_times(self) -> np.ndarray:
        return self.times[self.windowed_indices]

    @property
    def n(self) -> int:
        return len(self.times)

    @property
    def dt(self) -> float:
        return self.times[1] - self.times[0]

    @property
    def yerr(self) -> np.ndarray:
        if self.poisson_data:
            return np.sqrt(self.mean_model.get_value(self.times))
        else:
            return np.ones(self.n)

    @property
    def windowed_yerr(self) -> np.ndarray:
        return self.yerr[self.windowed_indices]

    @property
    def injection_id(self) -> str:
        return self._injection_id

    @injection_id.setter
    def injection_id(self, injection_id: Union[int, str]) -> None:
        self._injection_id = str(injection_id).zfill(2)

    @property
    def params(self) -> dict:
        self.update_params()
        return self._params

    @params.setter
    def params(self, params: dict) -> None:
        if isinstance(params, bilby.core.prior.PriorDict):
            self._params = params.sample()
        else:
            self._params = params

    @property
    def params_mean(self) -> dict:
        params_mean = dict()
        for param, value in self.params.items():
            if param.startswith("mean:"):
                params_mean[param.replace("mean:", "")] = value
        return params_mean

    @property
    def params_kernel(self) -> dict:
        params_kernel = dict()
        for param, value in self.params.items():
            if param.startswith("kernel:"):
                params_kernel[param.replace("kernel:", "")] = value
        if self.injection_mode == "qpo":
            params_kernel["log_b"] = -15
        return params_kernel

    def update_params(self) -> None:
        """ Sets the parameters in the GP safely. """
        for param, value in self._params.items():
            try:
                self.gp.set_parameter(param, value)
            except (ValueError, AttributeError):
                continue

    @property
    def cov(self) -> np.ndarray:
        self.gp.compute(self.windowed_times, self.windowed_yerr)
        return self.gp.get_matrix()

    @property
    def y(self) -> np.ndarray:
        self.gp.compute(self.windowed_times, self.windowed_yerr)
        return self.gp.get_value()

    @property
    def outdir_path_stub(self) -> str:
        return f"{self.outdir}/{self.injection_mode}/{self.likelihood_model}/{self.injection_id}"

    def save(self) -> None:
        """ Saves the injection data. """
        np.savetxt(f"{self.outdir_path_stub}_data.txt",
                   np.array([self.times, self.y_realisation, self.yerr]).T)
        with open(f"{self.outdir_path_stub}_params.json", "w") as f:
            json.dump(self.params, f)

    def plot(self) -> None:
        """ Plots the data with the true parameters fitted. """
        color = "#ff7f0e"
        plt.errorbar(self.times, self.y_realisation, yerr=self.yerr, fmt=".k", capsize=0, label="data")
        if self.injection_mode != "white_noise":
            self.update_params()
            x = np.linspace(self.windowed_times[0], self.windowed_times[-1], 5000)
            self.gp.compute(self.windowed_times, self.windowed_yerr)
            if self.likelihood_model == "celerite_windowed":
                plt.axvline(self.params["window_minimum"], color="cyan", label="start/end stochastic process")
                plt.axvline(self.params["window_maximum"], color="cyan")
            pred_mean, pred_var = self.gp.predict(self.y_realisation[self.windowed_indices], x, return_var=True)
            pred_std = np.sqrt(pred_var)
            plt.plot(x, pred_mean, color=color, label="Prediction")
            plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std,
                             color=color, alpha=0.3, edgecolor="none")

        plt.plot(self.times, self.gp.mean.get_value(self.times), color="green", label="Mean function")
        plt.xlabel("time [s]")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.outdir_path_stub}_data.pdf")
        plt.show()
        plt.clf()


def create_injection(
        params: dict, injection_mode: str, sampling_frequency: float = None, segment_length: float = None,
        times: np.ndarray = None, outdir: str = "injection_files", injection_id: Union[int, str] = 0,
        plot: bool = False, likelihood_model: str = "celerite", mean_model: Union[str, float] = "mean",
        n_components: int = 1, poisson_data: bool = False) -> None:
    """ Primary function to create arbitrary simulated data sets.

    Parameters
    ----------
    params:
        The parameters for the injection data.
    injection_mode:
        The kernel to use in the injection. Must be from `QPOEstimation.MODES`.
    sampling_frequency:
        The sampling frequency. Optional, only relevant if `times` is None.
    segment_length:
        The segment length. Optional, only relevant if `times` is None.
    times:
        The times. `sampling_frequency` and `segment_length` are easier to use if we want a
        equally spaced data set. We can choose arbitrary spacings by passing `times`.
    outdir:
        The directory to write the files into. (Default_value = 'injection_files').
    injection_id:
        The ID number of this data set.
    plot:
        Whether to create a plot of the data. (Default_value = False).
    likelihood_model:
        The likelihood to use. Must be from `QPOEstimation.LIKELIHOOD_MODELS`. (Default_value = 'celerite')
    mean_model:
        The mean model to use. Must be from one of the functions in `QPOEstimation.model.mean`, or 'mean'
        or a number for a constant mean model.
    n_components:
        The number of mean model flares to use. (Default_value = 1)
    poisson_data:
        Whether to create Poisson data.
        This assumes the underlying true rate and creates Poisson process data from it.
    """
    injection_creator = InjectionCreator(params=params, injection_mode=injection_mode,
                                         sampling_frequency=sampling_frequency, segment_length=segment_length,
                                         times=times, outdir=outdir, injection_id=injection_id,
                                         likelihood_model=likelihood_model, mean_model=mean_model,
                                         n_components=n_components, poisson_data=poisson_data)
    if outdir is not None:
        injection_creator.save()
    if plot:
        injection_creator.plot()
