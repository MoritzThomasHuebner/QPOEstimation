import celerite.terms
import george.kernels
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import bilby
import pandas as pd
from bilby.core.likelihood import CeleriteLikelihood, GeorgeLikelihood
from typing import Union

from QPOEstimation.parse import OSCILLATORY_MODELS
from QPOEstimation.utils import MetaDataAccessor
from QPOEstimation.likelihood import WindowedCeleriteLikelihood, get_kernel, get_mean_model

style_file = f"{Path(__file__).parent.absolute()}/paper.mplstyle"

_likelihood_classes = \
    Union[bilby.likelihood.CeleriteLikelihood, bilby.likelihood.GeorgeLikelihood, WindowedCeleriteLikelihood]


class GPResult(bilby.result.Result):

    kernel_type = MetaDataAccessor("kernel_type")
    jitter_term = MetaDataAccessor("jitter_term")
    mean_model = MetaDataAccessor("mean_model")
    n_components = MetaDataAccessor("n_components", default=1)
    times = MetaDataAccessor("times")
    y = MetaDataAccessor("y")
    yerr = MetaDataAccessor("yerr")
    likelihood_model = MetaDataAccessor("likelihood_model", default="celerite")
    truths = MetaDataAccessor("truths")
    offset = MetaDataAccessor("offset")

    def __init__(self, **kwargs):
        """ An extension to the standard `bilby` result.
        Implements a number of features that the regular `bilby` result can not do itself.
        Saves meta information about the data and analysis settings in `meta_data` and provides
        property-like access using the descriptors at the top of the class.

        Parameters
        ----------
        kwargs: Any args/kwargs that the regular `bilby` result takes
        """
        super().__init__(**kwargs)

    @property
    def corner_outdir(self) -> str:
        return self.outdir.replace("/results", "/corner")

    @property
    def fits_outdir(self) -> str:
        return self.outdir.replace("/results", "/fits")

    @property
    def max_likelihood_parameters(self) -> pd.Series:
        return self.posterior.iloc[-1]

    def get_random_posterior_samples(self, n_samples: int = 10) -> list:
        """ Returns a list of random posterior samples.

        Parameters
        ----------
        n_samples:
            The number of random samples  (Default_value = 10).

        Returns
        -------
        The samples.
        """
        return [self.posterior.iloc[np.random.randint(len(self.posterior))] for _ in range(n_samples)]

    def get_likelihood(self) -> _likelihood_classes:
        """ Reconstructs the likelihood instance based on the `meta_data` information.

        Returns
        -------
        The instance of the likelihood class used during the inference process.
        """
        if self.likelihood_model == "celerite_windowed":
            likelihood = WindowedCeleriteLikelihood(mean_model=self.get_mean_model(), kernel=self.get_kernel(),
                                                    t=self.times, y=self.y, yerr=self.yerr)
        elif self.likelihood_model == "celerite":
            likelihood = CeleriteLikelihood(mean_model=self.get_mean_model(), kernel=self.get_kernel(),
                                            t=self.times, y=self.y, yerr=self.yerr)
        elif self.likelihood_model == "george":
            likelihood = GeorgeLikelihood(mean_model=self.get_mean_model(), kernel=self.get_kernel(),
                                          t=self.times, y=self.y, yerr=self.yerr)
        else:
            raise ValueError
        return self._set_likelihood_parameters(likelihood=likelihood, parameters=self.max_likelihood_parameters)

    @staticmethod
    def _set_likelihood_parameters(
            likelihood: _likelihood_classes, parameters: Union[dict, pd.Series]) -> _likelihood_classes:
        """ Sets the parameters for the likelihood.

        Parameters
        ----------
        likelihood:
            The likelihood class instance.
        parameters:
            The parameters in a dict-like format.

        Returns
        -------
        The likelihood with the set parameters.
        """
        for name, value in parameters.items():
            try:
                likelihood.gp.set_parameter(name=name, value=value)
            except ValueError:
                pass
            try:
                likelihood.mean_model.set_parameter(name=name, value=value)
            except (ValueError, AttributeError):
                pass
            try:
                likelihood.parameters[name] = value
            except (ValueError, AttributeError):
                pass
        return likelihood

    def get_kernel(self) -> Union[celerite.terms.Term, george.kernels.Kernel]:
        """ Small wrapper for getting the kernel based on the `meta_data` information.

        Returns
        -------
        The kernel used during the inference process.
        """
        return get_kernel(kernel_type=self.kernel_type)

    def get_mean_model(self) -> Union[celerite.modeling.Model, george.modeling.Model]: # noqa
        """ Small wrapper for getting the mean model based on the `meta_data` information.

        Returns
        -------
        The mean model used during the inference process.
        """
        return get_mean_model(
            model_type=self.mean_model, n_components=self.n_components, y=self.y, offset=self.offset)

    @property
    def sampling_frequency(self) -> float:
        return 1 / (self.times[1] - self.times[0])

    @property
    def segment_length(self) -> float:
        return self.times[-1] - self.times[0]

    def plot_max_likelihood_psd(self, paper_style: bool = True, show: bool = True) -> None:
        """ Plots the maximum likelihood psd.

        Parameters
        ----------
        paper_style: Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        if paper_style:
            plt.style.use(style_file)
        Path(self.fits_outdir).mkdir(parents=True, exist_ok=True)
        likelihood = self.get_likelihood()
        psd_freqs = np.linspace(1/self.segment_length, self.sampling_frequency, 5000)
        psd = likelihood.gp.kernel.get_psd(psd_freqs * 2 * np.pi)

        plt.loglog(psd_freqs, psd, label="complete GP")
        for i, k in enumerate(likelihood.gp.kernel.terms):
            plt.loglog(psd_freqs, k.get_psd(psd_freqs * 2 * np.pi), "--", label=f"term {i}")

        plt.xlim(psd_freqs[0], psd_freqs[-1])
        plt.xlabel("f[Hz]")
        plt.ylabel("$S(f)$")
        plt.legend()
        try:
            plt.tight_layout()
        except Exception:
            pass
        plt.savefig(f"{self.fits_outdir}/{self.label}_psd.png")
        if show:
            plt.show()
        plt.clf()

    def plot_kernel(self, paper_style: bool = True, show: bool = True) -> None:
        """ Plots the maximum likelihood kernel function.

        Parameters
        ----------
        paper_style: Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        if paper_style:
            plt.style.use(style_file)
        Path(self.fits_outdir).mkdir(parents=True, exist_ok=True)
        likelihood = self.get_likelihood()
        taus = np.linspace(-0.5 * self.segment_length, 0.5 * self.segment_length, 1000)
        plt.plot(taus, likelihood.gp.kernel.get_value(taus), color="blue")
        plt.xlabel("tau [s]")
        plt.ylabel("kernel")
        try:
            plt.tight_layout()
        except Exception:
            pass
        plt.savefig(f"{self.fits_outdir}/{self.label}_max_like_kernel.pdf")
        if show:
            plt.show()
        plt.clf()

    def plot_lightcurve(
            self, start_time: float = None, end_time: float = None,
            paper_style: bool = True, show: bool = True) -> None:
        """ Plots the lightcurve and the maximum likelihood fit.

        Parameters
        ----------
        start_time:
            The start time from which to plot the data/fit.
        end_time:
            The start time up to which to plot the data/fit.
        paper_style:
            Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        if paper_style:
            plt.style.use(style_file)
        if start_time is None:
            start_time = self.times[0]
        if end_time is None:
            end_time = self.times[-1]
        Path(self.fits_outdir).mkdir(parents=True, exist_ok=True)
        likelihood = self.get_likelihood()

        jitter = 0
        for k in list(self.max_likelihood_parameters.keys()):
            if "log_sigma" in k and self.jitter_term:
                jitter = np.exp(self.max_likelihood_parameters[k])

        if self.likelihood_model == "celerite_windowed":
            plt.axvline(self.max_likelihood_parameters["window_minimum"], color="cyan",
                        label="$t_{\mathrm{start/end}}$")
            plt.axvline(self.max_likelihood_parameters["window_maximum"], color="cyan")
            x = np.linspace(self.max_likelihood_parameters["window_minimum"],
                            self.max_likelihood_parameters["window_maximum"], 5000)
            windowed_indices = np.where(
                np.logical_and(self.max_likelihood_parameters["window_minimum"] < self.times,
                               self.times < self.max_likelihood_parameters["window_maximum"]))
            likelihood.gp.compute(self.times[windowed_indices], self.yerr[windowed_indices] + jitter)
            pred_mean, pred_var = likelihood.gp.predict(self.y[windowed_indices], x, return_var=True)
        else:
            likelihood.gp.compute(self.times, self.yerr + jitter)
            x = np.linspace(start_time, end_time, 5000)
            pred_mean, pred_var = likelihood.gp.predict(self.y, x, return_var=True)
        pred_std = np.sqrt(pred_var)

        color = "#ff7f0e"
        plt.errorbar(self.times, self.y, yerr=self.yerr + jitter, fmt=".k", capsize=0, label="data")
        plt.plot(x, pred_mean, color=color, label="Prediction")
        plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
                         edgecolor="none")
        if self.mean_model != "mean":
            x = np.linspace(start_time, end_time, 5000)
            if isinstance(likelihood.mean_model, (float, int)):
                trend = np.ones(len(x)) * likelihood.mean_model
            else:
                trend = likelihood.mean_model.get_value(x)
            plt.plot(x, trend, color="green", label="Mean")
            samples = self.get_random_posterior_samples(10)
            for sample in samples:
                likelihood = self._set_likelihood_parameters(likelihood=likelihood, parameters=sample)
                if isinstance(likelihood.mean_model, (float, int)):
                    trend = np.ones(len(x)) * likelihood.mean_model
                else:
                    trend = likelihood.mean_model.get_value(x)
                plt.plot(x, trend, color="green", alpha=0.3)

        plt.xlabel("time [s]")
        plt.ylabel("Normalised flux")
        plt.legend(ncol=2)
        try:
            plt.tight_layout()
        except Exception:
            pass
        plt.savefig(f"{self.fits_outdir}/{self.label}_max_like_fit.pdf")
        if show:
            plt.show()
        plt.clf()

    def plot_residual(
            self, start_time: float = None, end_time: float = None,
            paper_style: bool = True, show: bool = True) -> None:
        """ Plots the lightcurve minus the maximum likelihood mean model and the maximum likelihood prediction.

        Parameters
        ----------
        start_time:
            The start time from which to plot the data/fit.
        end_time:
            The start time up to which to plot the data/fit.
        paper_style:
            Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        if paper_style:
            plt.style.use(style_file)
        if start_time is None:
            start_time = self.times[0]
        if end_time is None:
            end_time = self.times[-1]
        Path(self.fits_outdir).mkdir(parents=True, exist_ok=True)
        likelihood = self.get_likelihood()

        jitter = 0
        for k in list(self.max_likelihood_parameters.keys()):
            if "log_sigma" in k and self.jitter_term:
                jitter = np.exp(self.max_likelihood_parameters[k])

        if self.likelihood_model == "celerite_windowed":
            plt.axvline(self.max_likelihood_parameters["window_minimum"], color="cyan",
                        label="start/end stochastic process")
            plt.axvline(self.max_likelihood_parameters["window_maximum"], color="cyan")
            x = np.linspace(self.max_likelihood_parameters["window_minimum"],
                            self.max_likelihood_parameters["window_maximum"], 5000)
            windowed_indices = np.where(
                np.logical_and(self.max_likelihood_parameters["window_minimum"] < self.times,
                               self.times < self.max_likelihood_parameters["window_maximum"]))
            likelihood.gp.compute(self.times[windowed_indices], self.yerr[windowed_indices] + jitter)
            pred_mean, pred_var = likelihood.gp.predict(self.y[windowed_indices], x, return_var=True)
        else:
            likelihood.gp.compute(self.times, self.yerr + jitter)
            x = np.linspace(start_time, end_time, 5000)
            pred_mean, pred_var = likelihood.gp.predict(self.y, x, return_var=True)
        pred_std = np.sqrt(pred_var)

        if self.mean_model != "mean":
            x = np.linspace(start_time, end_time, 5000)
            if isinstance(likelihood.mean_model, (float, int)):
                trend = np.ones(len(self.times)) * likelihood.mean_model
                trend_fine = np.ones(len(x)) * likelihood.mean_model
            else:
                trend = likelihood.mean_model.get_value(self.times)
                trend_fine = likelihood.mean_model.get_value(x)
        else:
            trend = 0
            trend_fine = 0

        color = "#ff7f0e"
        plt.errorbar(self.times, self.y - trend, yerr=self.yerr + jitter, fmt=".k", capsize=0, label="data", color="black")
        plt.plot(x, pred_mean - trend_fine, color=color, label="Prediction")
        plt.fill_between(x, pred_mean + pred_std - trend_fine, pred_mean - pred_std - trend_fine, color=color,
                         alpha=0.3, edgecolor="none")

        plt.xlabel("time [s]")
        plt.ylabel("y")
        plt.legend()
        try:
            plt.tight_layout()
        except Exception:
            pass
        plt.savefig(f"{self.fits_outdir}/{self.label}_max_like_residual_fit.pdf")
        if show:
            plt.show()
        plt.clf()

    def plot_corner(self, show: bool = True, **kwargs) -> None:
        """ Corner plotting utility. Calls to `bilby` implemented `plot_corner`.

        Parameters
        ----------
        kwargs: All keyword arguments the bilby `plot_corner` method takes except for `outdir` and `truths`.
        """
        try:
            super().plot_corner(outdir=self.corner_outdir, truths=self.truths, **kwargs)
        except Exception:
            super().plot_corner(outdir=self.corner_outdir, **kwargs)
        if show:
            plt.show()

    def plot_frequency_posterior(self, paper_style: bool = True, show: bool = True) -> None:
        """ Plots the frequency posterior.

        Parameters
        ----------
        paper_style: Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        if paper_style:
            plt.style.use(style_file)

        Path(self.corner_outdir).mkdir(parents=True, exist_ok=True)
        if self.kernel_type in OSCILLATORY_MODELS:
            if "kernel:log_f" in self.posterior:
                frequency_samples = np.exp(np.array(self.posterior["kernel:log_f"]))
            elif "kernel:terms[0]:log_f" in self.posterior:
                frequency_samples = np.exp(np.array(self.posterior["kernel:terms[0]:log_f"]))
            else:
                return
            plt.hist(frequency_samples, bins="fd", density=True)
            plt.xlabel("frequency [Hz]")
            plt.ylabel("Normalised PDF")
            median = np.median(frequency_samples)
            percentiles = np.percentile(frequency_samples, [16, 84])
            plt.title(
                f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            try:
                plt.tight_layout()
            except Exception:
                pass
            plt.savefig(f"{self.corner_outdir}/{self.label}_frequency_posterior.pdf")
            if show:
                plt.show()
            plt.clf()
        elif self.kernel_type == "double_qpo":
            frequency_samples_1 = np.exp(np.array(self.posterior["kernel:terms[0]:log_f"]))
            frequency_samples_2 = np.exp(np.array(self.posterior["kernel:terms[1]:log_f"]))

            for i, frequency_samples in enumerate([frequency_samples_1, frequency_samples_2]):
                plt.hist(frequency_samples, bins="fd", density=True)
                plt.xlabel("frequency [Hz]")
                plt.ylabel("Normalised PDF")
                median = np.median(frequency_samples)
                percentiles = np.percentile(frequency_samples, [16, 84])
                plt.title(
                    f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
                try:
                    plt.tight_layout()
                except Exception:
                    pass
                plt.savefig(f"{self.corner_outdir}/{self.label}_frequency_posterior_{i}.pdf")
                if show:
                    plt.show()
                plt.clf()

    def plot_period_posterior(self, paper_style: bool = True, show: bool = True) -> None:
        """ Plots the period posterior.

        Parameters
        ----------
        paper_style: Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        if paper_style:
            plt.style.use(style_file)
        Path(self.corner_outdir).mkdir(parents=True, exist_ok=True)
        if self.kernel_type in OSCILLATORY_MODELS:
            if "kernel:log_f" in self.posterior:
                period_samples = 1/np.exp(np.array(self.posterior["kernel:log_f"]))
            elif "kernel:terms[0]:log_f" in self.posterior:
                period_samples = 1/np.exp(np.array(self.posterior["kernel:terms[0]:log_f"]))
            else:
                return
            plt.hist(period_samples, bins="fd", density=True)
            plt.xlabel("period [s]")
            # plt.xlim(6.5, 10.5)
            plt.ylabel("Normalised PDF")
            median = np.median(period_samples)
            percentiles = np.percentile(period_samples, [16, 84])
            plt.title(
                f"{np.mean(period_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            try:
                plt.tight_layout()
            except Exception:
                pass
            plt.savefig(f"{self.corner_outdir}/{self.label}_period_posterior.pdf")
            if show:
                plt.show()
            plt.clf()

    def plot_qpo_log_amplitude(self, paper_style: bool = True, show: bool = True) -> None:
        """ Plots the QPO log amplitude posterior.

        Parameters
        ----------
        paper_style: Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        if self.kernel_type == "qpo_plus_red_noise":
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            if paper_style:
                plt.style.use(style_file)
            Path(self.corner_outdir).mkdir(parents=True, exist_ok=True)
            label = "kernel:terms[0]:log_a"
            log_amplitude_samples = np.array(self.posterior[label])
            plt.hist(log_amplitude_samples, bins="fd", density=True)
            plt.xlabel("$\ln \,a$")
            plt.ylabel("Normalised PDF")
            median = np.median(log_amplitude_samples)
            percentiles = np.percentile(log_amplitude_samples, [16, 84])
            plt.title(
                f"{np.mean(log_amplitude_samples):.2f} + "
                f"{percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            try:
                plt.tight_layout()
            except Exception:
                pass
            plt.savefig(f"{self.corner_outdir}/{self.label}_log_amplitude_posterior.pdf")
            if show:
                plt.show()
            plt.clf()

    def plot_amplitude_ratio(self, paper_style: bool = True, show: bool = True) -> None:
        """ Plots the amplitude ratio posterior.

        Parameters
        ----------
        paper_style: Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        if self.kernel_type == "qpo_plus_red_noise":
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            if paper_style:
                plt.style.use(style_file)
            Path(self.corner_outdir).mkdir(parents=True, exist_ok=True)
            qpo_log_amplitude_samples = np.array(self.posterior["kernel:terms[0]:log_a"])
            red_noise_log_amplitude_samples = np.array(self.posterior["kernel:terms[1]:log_a"])
            amplitude_ratio_samples = np.exp(qpo_log_amplitude_samples - red_noise_log_amplitude_samples)
            plt.hist(amplitude_ratio_samples, bins="fd", density=True)
            plt.xlabel("$a_{\mathrm{qpo}}/a_{\mathrm{rn}}$")
            plt.ylabel("Normalised PDF")
            median = np.median(amplitude_ratio_samples)
            percentiles = np.percentile(amplitude_ratio_samples, [16, 84])
            plt.title(
                f"{np.mean(amplitude_ratio_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            try:
                plt.tight_layout()
            except Exception:
                pass
            plt.savefig(f"{self.corner_outdir}/{self.label}_amplitude_ratio_posterior.pdf")
            if show:
                plt.show()
            plt.clf()

    def plot_log_red_noise_power(self, paper_style: bool = True, show: bool = True) -> None:
        """ Plots red noise power posterior. This is not very well reasoned and we did not implement it in our paper.

        Parameters
        ----------
        paper_style: Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        if self.kernel_type == "qpo_plus_red_noise":
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            if paper_style:
                plt.style.use(style_file)
            Path(self.corner_outdir).mkdir(parents=True, exist_ok=True)
            log_a_samples = np.array(self.posterior["kernel:terms[1]:log_a"])
            log_c_samples = np.array(self.posterior["kernel:terms[1]:log_c"])
            power_samples = np.log(power_red_noise(a=np.exp(log_a_samples), c=np.exp(log_c_samples)))
            plt.hist(power_samples, bins="fd", density=True)
            plt.xlabel("$\ln P_{\mathrm{rn}}$")
            plt.ylabel("Normalised PDF")
            median = np.median(power_samples)
            percentiles = np.percentile(power_samples, [16, 84])
            plt.title(
                f"{np.mean(power_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            try:
                plt.tight_layout()
            except Exception:
                pass
            plt.savefig(f"{self.corner_outdir}/{self.label}_red_noise_power_samples.pdf")
            if show:
                plt.show()
            plt.clf()

    def plot_log_qpo_power(self, paper_style: bool = True, show: bool = True) -> None:
        """ Plots the QPO log power posterior. This is not very well reasoned and we did not implement it in our paper.

        Parameters
        ----------
        paper_style: Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        if self.kernel_type == "qpo_plus_red_noise":
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            if paper_style:
                plt.style.use(style_file)
            Path(self.corner_outdir).mkdir(parents=True, exist_ok=True)
            log_a_samples = np.array(self.posterior["kernel:terms[0]:log_a"])
            log_c_samples = np.array(self.posterior["kernel:terms[0]:log_c"])
            log_f_samples = np.array(self.posterior["kernel:terms[0]:log_f"])
            power_samples = np.log(power_qpo(a=np.exp(log_a_samples), c=np.exp(log_c_samples), f=np.exp(log_f_samples)))
            plt.hist(power_samples, bins="fd", density=True)
            plt.xlabel("$\ln P_{\mathrm{qpo}}$")
            plt.ylabel("Normalised PDF")
            median = np.median(power_samples)
            percentiles = np.percentile(power_samples, [16, 84])
            plt.title(
                f"{np.mean(power_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            try:
                plt.tight_layout()
            except Exception:
                pass
            plt.savefig(f"{self.corner_outdir}/{self.label}_qpo_power_samples.pdf")
            if show:
                plt.show()
            plt.clf()

    def plot_duration_posterior(self, paper_style: bool = True, show: bool = True) -> None:
        """ Plots the duration posterior for the `celerite_windowed` likelihood class.

        Parameters
        ----------
        paper_style: Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plot.
        """
        if self.likelihood_model == "celerite_windowed":
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            if paper_style:
                plt.style.use(style_file)
            Path(self.corner_outdir).mkdir(parents=True, exist_ok=True)
            t_start_samples = np.array(self.posterior["window_minimum"])
            t_end_samples = np.array(self.posterior["window_maximum"])
            duration_samples = t_end_samples - t_start_samples

            plt.hist(duration_samples, bins="fd", density=True)
            plt.xlabel("duration [s]")
            plt.ylabel("Normalised PDF")
            median = np.median(duration_samples)
            percentiles = np.percentile(duration_samples, [16, 84])
            plt.title(
                f"{np.mean(duration_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            try:
                plt.tight_layout()
            except Exception:
                pass
            plt.savefig(f"{self.corner_outdir}/{self.label}_duration_posterior.pdf")
            if show:
                plt.show()
            plt.clf()

    def plot_all(self, paper_style=True, show: bool = True):
        """ Plots all relevant posteriors.

        Parameters
        ----------
        paper_style: Whether to use the `paper.mplstyle` style file.
        show: Whether to show the plots.
        """
        self.plot_corner(show=show)
        try:
            self.plot_max_likelihood_psd(paper_style=paper_style, show=show)
        except Exception:
            pass
        try:
            self.plot_kernel(paper_style=paper_style, show=show)
        except Exception:
            pass
        self.plot_lightcurve(paper_style=paper_style, show=show)
        self.plot_residual(paper_style=paper_style, show=show)
        self.plot_frequency_posterior(paper_style=paper_style, show=show)
        self.plot_period_posterior(paper_style=paper_style, show=show)
        self.plot_duration_posterior(paper_style=paper_style, show=show)
        # self.plot_qpo_log_amplitude(paper_style=paper_style, show=show)
        # self.plot_log_red_noise_power(paper_style=paper_style, show=show)
        # self.plot_log_qpo_power(paper_style=paper_style, show=show)
        # self.plot_amplitude_ratio(paper_style=paper_style, show=show)


def power_qpo(
        a: Union[float, np.ndarray], c: Union[float, np.ndarray], f: Union[float, np.ndarray])\
        -> Union[float, np.ndarray]:
    """ Effect size of the QPO. This is not very well reasoned and we did not implement it in our paper.

    Returns
    -------
    The integral of the square of the kernel.
    """
    return a * np.sqrt((c**2 + 2 * np.pi**2 * f**2)/(c * (c**2 + 4 * np.pi**2 * f**2)))


def power_red_noise(a: Union[float, np.ndarray], c: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """ Effect size of the red noise. This is not very well reasoned and we did not implement it in our paper.

    Returns
    -------
    The integral of the square of the kernel.
    """
    return a / c**0.5
