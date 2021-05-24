import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import bilby

from QPOEstimation.utils import MetaDataAccessor
from QPOEstimation.likelihood import CeleriteLikelihood, WindowedCeleriteLikelihood, get_kernel, get_mean_model
from QPOEstimation.model.celerite import power_qpo, power_red_noise

OSCILLATORY_MODELS = ["qpo", "pure_qpo", "general_qpo"]


class GPResult(bilby.result.Result):

    kernel_type = MetaDataAccessor('kernel_type')
    jitter_term = MetaDataAccessor('jitter_term')
    mean_model = MetaDataAccessor('mean_model')
    n_components = MetaDataAccessor('n_components', default=1)
    times = MetaDataAccessor('times')
    y = MetaDataAccessor('y')
    yerr = MetaDataAccessor('yerr')
    likelihood_model = MetaDataAccessor('likelihood_model', default='gaussian_process')
    truths = MetaDataAccessor('truths')
    offset = MetaDataAccessor('offset')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def corner_outdir(self):
        return self.outdir.replace('results', 'corner')

    @property
    def fits_outdir(self):
        return self.outdir.replace('results', 'fits')

    @property
    def max_likelihood_parameters(self):
        return self.posterior.iloc[-1]

    def get_random_posterior_samples(self, n_samples=10):
        return [self.posterior.iloc[np.random.randint(len(self.posterior))] for _ in range(n_samples)]

    def get_likelihood(self):
        if self.likelihood_model == "gaussian_process_windowed":
            likelihood = WindowedCeleriteLikelihood(mean_model=self.get_mean_model(), kernel=self.get_kernel(),
                                                    fit_mean=True, t=self.times, y=self.y, yerr=self.yerr)
        elif self.likelihood_model == 'gaussian_process':
            likelihood = CeleriteLikelihood(mean_model=self.get_mean_model(), kernel=self.get_kernel(), fit_mean=True,
                                            t=self.times, y=self.y, yerr=self.yerr)
        else:
            raise ValueError
        return self._set_likelihood_parameters(likelihood=likelihood, parameters=self.max_likelihood_parameters)

    @staticmethod
    def _set_likelihood_parameters(likelihood, parameters):
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

    def get_kernel(self):
        return get_kernel(kernel_type=self.kernel_type)

    def get_mean_model(self):
        mean_model, _ = get_mean_model(model_type=self.mean_model, n_components=self.n_components, y=self.y,
                                       offset=self.offset)
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
        psd_freqs = np.linspace(1/self.segment_length, self.sampling_frequency, 5000)
        psd = likelihood.gp.kernel.get_psd(psd_freqs * 2 * np.pi)

        plt.loglog(psd_freqs, psd, label='complete GP')
        for i, k in enumerate(likelihood.gp.kernel.terms):
            plt.loglog(psd_freqs, k.get_psd(psd_freqs * 2 * np.pi), "--", label=f'term {i}')

        plt.xlim(psd_freqs[0], psd_freqs[-1])
        plt.xlabel("f[Hz]")
        plt.ylabel("$S(f)$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.fits_outdir}/{self.label}_psd.png")
        plt.clf()

    def plot_kernel(self):
        Path(self.fits_outdir).mkdir(parents=True, exist_ok=True)
        likelihood = self.get_likelihood()
        taus = np.linspace(-0.5*self.segment_length, 0.5*self.segment_length, 1000)
        plt.plot(taus, likelihood.gp.kernel.get_value(taus), color="blue")

        samples = self.get_random_posterior_samples(20)
        for sample in samples:
            likelihood = self._set_likelihood_parameters(likelihood=likelihood, parameters=sample)
            plt.plot(taus, likelihood.gp.kernel.get_value(taus), color="blue", alpha=0.3)
        plt.xlabel('tau [s]')
        plt.ylabel('kernel')
        plt.tight_layout()
        plt.savefig(f"{self.fits_outdir}/{self.label}_max_like_kernel.png")
        plt.clf()

    def plot_lightcurve(self, start_time=None, end_time=None):
        if start_time is None:
            start_time = self.times[0]
        if end_time is None:
            end_time = self.times[-1]
        Path(self.fits_outdir).mkdir(parents=True, exist_ok=True)
        likelihood = self.get_likelihood()

        jitter = 0
        for k in list(self.max_likelihood_parameters.keys()):
            if 'log_sigma' in k and self.jitter_term:
                jitter = np.exp(self.max_likelihood_parameters[k])

        if self.likelihood_model == 'gaussian_process_windowed':
            plt.axvline(self.max_likelihood_parameters['window_minimum'], color='cyan', label='start/end stochastic process')
            plt.axvline(self.max_likelihood_parameters['window_maximum'], color='cyan')
            x = np.linspace(self.max_likelihood_parameters['window_minimum'], self.max_likelihood_parameters['window_maximum'], 5000)
            windowed_indices = np.where(
                np.logical_and(self.max_likelihood_parameters['window_minimum'] < self.times,
                               self.times < self.max_likelihood_parameters['window_maximum']))
            likelihood.gp.compute(self.times[windowed_indices], self.yerr[windowed_indices] + jitter)
            pred_mean, pred_var = likelihood.gp.predict(self.y[windowed_indices], x, return_var=True)
        else:
            likelihood.gp.compute(self.times, self.yerr + jitter)
            x = np.linspace(start_time, end_time, 5000)
            pred_mean, pred_var = likelihood.gp.predict(self.y, x, return_var=True)
        pred_std = np.sqrt(pred_var)

        color = "#ff7f0e"
        plt.errorbar(self.times, self.y, yerr=self.yerr + jitter, fmt=".k", capsize=0, label='data')
        plt.plot(x, pred_mean, color=color, label='Prediction')
        plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color, alpha=0.3,
                         edgecolor="none")
        if self.mean_model != "mean":
            x = np.linspace(start_time, end_time, 5000)
            trend = likelihood.mean_model.get_value(x)
            plt.plot(x, trend, color='green', label='Mean')
            samples = self.get_random_posterior_samples(10)
            for sample in samples:
                likelihood = self._set_likelihood_parameters(likelihood=likelihood, parameters=sample)
                trend = likelihood.mean_model.get_value(x)
                plt.plot(x, trend, color='green', alpha=0.3)

        plt.xlabel("time [s]")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.fits_outdir}/{self.label}_max_like_fit.png")
        plt.show()
        plt.clf()

    def plot_corner(self, **kwargs):
        super().plot_corner(outdir=self.corner_outdir, truths=self.truths, **kwargs)

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
            plt.savefig(f"{self.corner_outdir}/{self.label}_frequency_posterior.png")
            plt.clf()

    def plot_qpo_log_amplitude(self):
        if self.kernel_type == "general_qpo":
            label = 'kernel:terms[0]:log_a'
        elif self.kernel_type == "pure_qpo":
            label = 'kernel:log_a'
        else:
            return
        log_amplitude_samples = np.array(self.posterior[label])
        plt.hist(log_amplitude_samples, bins="fd", density=True)
        plt.xlabel('$\ln \,a$')
        plt.ylabel('normalised PDF')
        median = np.median(log_amplitude_samples)
        percentiles = np.percentile(log_amplitude_samples, [16, 84])
        plt.title(
            f"{np.mean(log_amplitude_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
        plt.tight_layout()
        plt.savefig(f"{self.corner_outdir}/{self.label}_log_amplitude_posterior.png")
        plt.clf()

    def plot_amplitude_ratio(self):
        if self.kernel_type == "general_qpo":
            qpo_log_amplitude_samples = np.array(self.posterior['kernel:terms[0]:log_a'])
            red_noise_log_amplitude_samples = np.array(self.posterior['kernel:terms[1]:log_a'])
            amplitude_ratio_samples = np.exp(qpo_log_amplitude_samples - red_noise_log_amplitude_samples)
            plt.hist(amplitude_ratio_samples, bins="fd", density=True)
            plt.xlabel('$a_{\mathrm{qpo}}/a_{\mathrm{rn}}$')
            plt.ylabel('normalised PDF')
            median = np.median(amplitude_ratio_samples)
            percentiles = np.percentile(amplitude_ratio_samples, [16, 84])
            plt.title(
                f"{np.mean(amplitude_ratio_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            plt.tight_layout()
            plt.savefig(f"{self.corner_outdir}/{self.label}_amplitude_ratio_posterior.png")
            plt.clf()

    def plot_log_red_noise_power(self):
        if self.kernel_type == "general_qpo":
            log_a_samples = np.array(self.posterior['kernel:terms[1]:log_a'])
            log_c_samples = np.array(self.posterior['kernel:terms[1]:log_c'])
            power_samples = np.log(power_red_noise(a=np.exp(log_a_samples), c=np.exp(log_c_samples)))
            plt.hist(power_samples, bins="fd", density=True)
            plt.xlabel('$P_{\mathrm{rn}}$')
            plt.ylabel('normalised PDF')
            median = np.median(power_samples)
            percentiles = np.percentile(power_samples, [16, 84])
            plt.title(
                f"{np.mean(power_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            plt.tight_layout()
            plt.savefig(f"{self.corner_outdir}/{self.label}_red_noise_power_samples.png")
            plt.clf()

    def plot_log_qpo_power(self):
        if self.kernel_type == "general_qpo":
            log_a_samples = np.array(self.posterior['kernel:terms[0]:log_a'])
            log_c_samples = np.array(self.posterior['kernel:terms[0]:log_c'])
            log_f_samples = np.array(self.posterior['kernel:terms[0]:log_f'])
            power_samples = np.log(power_qpo(a=np.exp(log_a_samples), c=np.exp(log_c_samples), f=np.exp(log_f_samples)))
            plt.hist(power_samples, bins="fd", density=True)
            plt.xlabel('$P_{\mathrm{qpo}}$')
            plt.ylabel('normalised PDF')
            median = np.median(power_samples)
            percentiles = np.percentile(power_samples, [16, 84])
            plt.title(
                f"{np.mean(power_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
            plt.tight_layout()
            plt.savefig(f"{self.corner_outdir}/{self.label}_qpo_power_samples.png")
            plt.clf()


    def plot_all(self):
        self.plot_max_likelihood_psd()
        self.plot_kernel()
        self.plot_frequency_posterior()
        self.plot_lightcurve()
        self.plot_qpo_log_amplitude()
        self.plot_log_red_noise_power()
        self.plot_log_qpo_power()
        self.plot_corner()
        # self.plot_amplitude_ratio()
