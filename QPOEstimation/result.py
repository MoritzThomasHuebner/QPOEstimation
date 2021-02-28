import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import bilby

from QPOEstimation.utils import MetaDataAccessor
from QPOEstimation.likelihood import CeleriteLikelihood, WindowedCeleriteLikelihood, get_kernel, get_mean_model

OSCILLATORY_MODELS = ["qpo", "pure_qpo", "general_qpo"]


class GPResult(bilby.result.Result):

    kernel_type = MetaDataAccessor('kernel_type')
    mean_model = MetaDataAccessor('mean_model')
    n_components = MetaDataAccessor('n_components')
    times = MetaDataAccessor('times')
    y = MetaDataAccessor('y')
    yerr = MetaDataAccessor('yerr')
    likelihood_model = MetaDataAccessor('likelihood_model')
    truths = MetaDataAccessor('truths')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def corner_outdir(self):
        return self.outdir.replace('result', 'corner')

    @property
    def fits_outdir(self):
        return self.outdir.replace('result', 'fits')

    @property
    def max_likelihood_parameters(self):
        return self.posterior.iloc[-1]

    def get_likelihood(self):
        if self.likelihood_model == "gaussian_process_windowed":
            likelihood = WindowedCeleriteLikelihood(mean_model=self.get_mean_model(), kernel=self.get_kernel(),
                                                    fit_mean=True, t=self.times, y=self.y, yerr=self.yerr)
        elif self.likelihood_model == 'gaussian_process':
            likelihood = CeleriteLikelihood(mean_model=self.get_mean_model(), kernel=self.get_kernel(), fit_mean=True,
                                            t=self.times, y=self.y, yerr=self.yerr)
        else:
            raise ValueError
        for name, value in self.max_likelihood_parameters.items():
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

    def get_kernel(self):
        return get_kernel(kernel_type=self.kernel_type)

    def get_mean_model(self):
        mean_model, _ = get_mean_model(model_type=self.mean_model, n_components=self.n_components, y=self.y)
        return mean_model

    @property
    def sampling_frequency(self):
        return 1 / (self.times[1] - self.times[0])

    @property
    def segment_length(self):
        return self.times[-1] - self.times[0]

    def plot_max_likelihood_psd(self):
        Path(self.fits_outdir).mkdir(parents=True, exist_ok=True)
        likelihood = self.get_likelihood()
        psd_freqs = np.linspace(1/self.segment_length, self.sampling_frequncy, 5000)
        psd = likelihood.gp.kernel.get_psd(psd_freqs * 2 * np.pi)

        plt.loglog(psd_freqs, psd, label='complete GP')
        for i, k in enumerate(likelihood.gp.kernel.terms):
            plt.loglog(psd_freqs, k.get_psd(psd_freqs * 2 * np.pi), "--", label=f'term {i}')

        plt.xlim(psd_freqs[0], psd_freqs[-1])
        plt.xlabel("f[Hz]")
        plt.ylabel("$S(f)$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.fits_outdir}/{self.label}_psd")
        plt.clf()

    def plot_kernel(self):
        Path(self.fits_outdir).mkdir(parents=True, exist_ok=True)
        likelihood = self.get_likelihood()
        taus = np.linspace(-0.5*self.segment_length, 0.5*self.segment_length, 1000)
        plt.plot(taus, likelihood.gp.kernel.get_value(taus))
        plt.xlabel('tau [s]')
        plt.ylabel('kernel')
        plt.tight_layout()
        plt.savefig(f"{self.fits_outdir}/{self.label}_max_like_kernel")
        plt.clf()

    def plot_lightcurve(self):
        Path(self.fits_outdir).mkdir(parents=True, exist_ok=True)
        likelihood = self.get_likelihood()
        if self.likelihood_model == 'gaussian_process_windowed':
            plt.axvline(self.max_likelihood_parameters['window_minimum'], color='cyan', label='start/end stochastic process')
            plt.axvline(self.max_likelihood_parameters['window_maximum'], color='cyan')
            x = np.linspace(self.max_likelihood_parameters['window_minimum'], self.max_likelihood_parameters['window_maximum'], 5000)
            windowed_indices = np.where(
                np.logical_and(self.max_likelihood_parameters['window_minimum'] < self.times,
                               self.times < self.max_likelihood_parameters['window_maximum']))
            likelihood.gp.compute(self.times[windowed_indices], np.sqrt(self.yerr[windowed_indices]))
            pred_mean, pred_var = likelihood.gp.predict(self.y[windowed_indices], x, return_var=True)
        else:
            x = np.linspace(self.times[0], self.times[-1], 5000)
            pred_mean, pred_var = likelihood.gp.predict(self.y, x, return_var=True)
        pred_std = np.sqrt(pred_var)

        color = "#ff7f0e"
        plt.errorbar(self.times, self.y, yerr=np.sqrt(self.yerr), fmt=".k", capsize=0, label='data')
        plt.plot(x, pred_mean, color=color, label='Prediction')
        plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
                         edgecolor="none")
        if self.mean_model != "mean":
            x = np.linspace(self.times[0], self.times[-1], 5000)
            trend = likelihood.mean_model.get_value(x)
            plt.plot(x, trend, color='green', label='Trend')

        plt.xlabel("time [s]")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.fits_outdir}/{self.label}_max_like_fit")
        plt.show()
        plt.clf()

    def plot_corner(self, **kwargs):
        super().plot_corner(outdir=self.corner_outdir, truths=self.truths)

    def plot_frequency_posterior(self):
        if self.kernel_type in OSCILLATORY_MODELS:
            if 'kernel:log_f' in self.posterior:
                frequency_samples = np.exp(np.array(self.posterior['kernel:log_f']))
            elif 'kernel:terms[0]:log_f' in self.posterior:
                frequency_samples = np.exp(np.array(self.posterior['kernel:terms[0]:log_f']))
            else:
                return
            plt.hist(frequency_samples, bins="fd", density=True)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('normalised PDF')
            median = np.median(frequency_samples)
            percentiles = np.percentile(frequency_samples, [16, 84])
            plt.title(
                f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            plt.tight_layout()
            plt.savefig(f"{self.corner_outdir}/{self.label}_frequency_posterior")
            plt.clf()

    def plot_all(self):
        self.plot_corner()
        self.plot_max_likelihood_psd()
        self.plot_kernel()
        self.plot_frequency_posterior()
        self.plot_lightcurve()
