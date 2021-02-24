import json
from pathlib import Path

import bilby
import numpy as np
import celerite
import matplotlib.pyplot as plt

from QPOEstimation.likelihood import get_kernel, get_mean_model
from QPOEstimation.utils import get_injection_outdir


class InjectionCreator(object):

    def __init__(self, params, injection_mode, sampling_frequency=256, segment_length=1., times=None,
                 outdir='injection_files', injection_id=0, likelihood_model='gaussian_process', mean_model=None,
                 n_components=1, poisson_data=False):
        self.params = params
        self.injection_mode = injection_mode
        self.sampling_frequency = sampling_frequency
        self.segment_length = segment_length
        self.outdir = outdir
        self.injection_id = str(injection_id).zfill(2)
        self.poisson_data = poisson_data
        self.create_outdir()
        self.likelihood_model = likelihood_model
        self.kernel = get_kernel(self.injection_mode)

        self.times = times
        self.mean_model = get_mean_model(model_type=mean_model, n_components=n_components, y=0)
        for key, value in params.items():
            self.mean_model.__setattr__(key.replace('mean:', ''), value)
        self.gp = celerite.GP(kernel=self.kernel, mean=self.mean_model)
        self.gp.compute(self.windowed_times, self.windowed_yerr)
        self.update_params()
        self.y_realisation = self.gp.mean.get_value(self.times)
        self.y_realisation[self.windowed_indices] = self.gp.sample()
        self.y_realisation[self.outside_window_indices] += \
            np.random.normal(size=len(self.outside_window_indices)) * self.yerr[self.outside_window_indices]

    def create_outdir(self):
        Path(f'{self.outdir}/{self.injection_mode}/{self.likelihood_model}').mkdir(exist_ok=True, parents=True)

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, times):
        if times is None:
            self._times = np.linspace(0, self.segment_length, int(self.sampling_frequency * self.segment_length))
        else:
            self._times = times

    @property
    def windowed_indices(self):
        if self.likelihood_model == 'gaussian_process':
            return np.where(np.logical_and(-np.inf < self.times, self.times < np.inf))[0]
        else:
            return np.where(np.logical_and(self.params['window_minimum'] < self.times,
                                           self.times < self.params['window_maximum']))[0]

    @property
    def outside_window_indices(self):
        if self.likelihood_model == 'gaussian_process_windowed':
            return np.where(np.logical_or(self.params['window_minimum'] >= self.times,
                                          self.times >= self.params['window_maximum']))[0]
        else:
            return []

    @property
    def windowed_times(self):
        return self.times[self.windowed_indices]

    @property
    def n(self):
        return len(self.times)

    @property
    def dt(self):
        return self.times[1] - self.times[0]

    @property
    def yerr(self):
        if self.poisson_data:
            return np.sqrt(self.mean_model.get_value(self.times))
        else:
            return np.ones(self.n)

    @property
    def windowed_yerr(self):
        return self.yerr[self.windowed_indices]

    @property
    def params(self):
        self.update_params()
        return self._params

    @params.setter
    def params(self, params):
        if isinstance(params, bilby.core.prior.PriorDict):
            self._params = params.sample()
        else:
            self._params = params

    @property
    def params_mean(self):
        params_mean = dict()
        for param, value in self.params.items():
            if param.startswith('mean:'):
                params_mean[param.replace('mean:', '')] = value
        return params_mean

    @property
    def params_kernel(self):
        params_kernel = dict()
        for param, value in self.params.items():
            if param.startswith('kernel:'):
                params_kernel[param.replace('kernel:', '')] = value
        if self.injection_mode == 'qpo':
            params_kernel['log_b'] = -15
        return params_kernel

    def update_params(self):
        for param, value in self._params.items():
            try:
                self.gp.set_parameter(param, value)
            except (ValueError, AttributeError):
                continue

    @property
    def cov(self):
        self.gp.compute(self.windowed_times, self.windowed_yerr)
        return self.gp.get_matrix()

    @property
    def y(self):
        self.gp.compute(self.windowed_times, self.windowed_yerr)
        return self.gp.get_value()

    @property
    def outdir_path(self):
        return f"{self.outdir}/{self.injection_mode}/{self.likelihood_model}/{self.injection_id}"

    def save(self):
        np.savetxt(f'{self.outdir_path}_data.txt',
                   np.array([self.times, self.y_realisation, self.yerr]).T)
        with open(f'{self.outdir_path}_params.json', 'w') as f:
            json.dump(self.params, f)

    def plot(self):
        color = "#ff7f0e"
        plt.errorbar(self.times, self.y_realisation, yerr=self.yerr, fmt=".k", capsize=0, label='data')
        if self.injection_mode != 'white_noise':
            self.update_params()
            x = np.linspace(self.windowed_times[0], self.windowed_times[-1], 5000)
            self.gp.compute(self.windowed_times, self.windowed_yerr)
            if self.likelihood_model == 'gaussian_process_windowed':
                plt.axvline(self.params['window_minimum'], color='cyan', label='start/end stochastic process')
                plt.axvline(self.params['window_maximum'], color='cyan')
            pred_mean, pred_var = self.gp.predict(self.y_realisation[self.windowed_indices], x, return_var=True)
            pred_std = np.sqrt(pred_var)
            plt.plot(x, pred_mean, color=color, label='Prediction')
            plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std,
                             color=color, alpha=0.3, edgecolor="none")

        plt.plot(self.times, self.gp.mean.get_value(self.times), color='green', label='Mean function')
        plt.legend()
        plt.savefig(f'{self.outdir_path}_data.pdf')
        plt.show()
        plt.clf()


def create_injection(params, injection_mode, sampling_frequency=None, segment_length=None, times=None,
                     outdir='injection_files', injection_id=0, plot=False, likelihood_model='gaussian_process'):
    injection_creator = InjectionCreator(params=params, injection_mode=injection_mode,
                                         sampling_frequency=sampling_frequency, segment_length=segment_length,
                                         times=times, outdir=outdir, injection_id=injection_id,
                                         likelihood_model=likelihood_model)

    injection_creator.save()
    if plot:
        injection_creator.plot()
