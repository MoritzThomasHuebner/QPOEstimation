import warnings

import bilby
from scipy.signal import periodogram

import QPOEstimation
from .plotting import *


class InjectionStudyPostProcessor(object):

    def __init__(self, start_times, end_times, durations, outdir, label, times, frequencies, normalisation, y,
                 outdir_noise_periodogram, outdir_qpo_periodogram, injection_parameters=None, injection_psds=None,
                 extension_mode=None):
        """ This class is used for post-processing steps for the results for the "Pitfalls of periodograms" paper. """
        self.ln_bfs = np.array([])
        self.log_frequency_spreads = np.array([])
        self.durations_reduced = np.array([])
        self.snrs_optimal = np.array([])
        self.snrs_max_like = np.array([])
        self.snrs_max_like_quantiles = []
        self.extension_factors = np.array([])
        self.delta_bics = np.array([])
        self.chi_squares = np.array([])
        self.chi_squares_qpo = np.array([])
        self.chi_squares_red_noise = np.array([])
        self.chi_squares_high_freqs = np.array([])

        self.start_times = start_times
        self.end_times = end_times
        self.durations = durations
        self.label = label
        self.outdir = outdir
        self.times = times
        self.frequencies = frequencies
        self.y = y
        self.normalisation = normalisation

        self.sampling_frequency = int(round(1/(self.times[1] - self.times[0])))
        self.outdir_noise_periodogram = outdir_noise_periodogram
        self.outdir_qpo_periodogram = outdir_qpo_periodogram
        self._freqs_combined_periodogram = None
        self._powers_combined_periodogram = None
        self._res_noise = None
        self._res_qpo = None
        self.injection_parameters = injection_parameters
        self.injection_psds = injection_psds
        self.extension_mode = extension_mode
        self.x_break = None
        self.x_break_max_like = None
        try:
            self._calculate_x_break()
        except KeyError as e:
            print(e)

    def _calculate_x_break(self):
        if self.injection_parameters is not None and self.extension_mode == "white_noise":
            self.x_break = \
                self.injection_parameters["beta"] / self.injection_parameters["sigma"] * \
                self.injection_parameters["central_frequency"] ** (-self.injection_parameters["alpha"])

    def _calculate_max_like_x_break(self):
        if self.extension_mode == "white_noise":
            self.x_break_max_like = \
                self._beta_max_like / self._white_noise_max_like * \
                self._central_frequency_max_like ** (-self._alpha_max_like)

    def plot_chi_squares(self, show=False):
        plot_chi_squares(outdir=self.outdir, label=self.label, extension_factors=self.extension_factors,
                         chi_squares=self.chi_squares, chi_squares_qpo=self.chi_squares_qpo,
                         chi_squares_red_noise=self.chi_squares_red_noise,
                         chi_squares_high_freqs=self.chi_squares_high_freqs, show=show)

    def plot_snrs(self, show=False):
        plot_snrs(
            outdir=self.outdir, label=self.label, extension_factors=self.extension_factors,
            snrs_optimal=self.snrs_optimal, snrs_max_like=self.snrs_max_like,
            snrs_max_like_quantiles=self.snrs_max_like_quantiles, x_break=self.x_break, show=show)

    def plot_ln_bfs(self, show=False):
        plot_ln_bfs(outdir=self.outdir, label=self.label, extension_factors=self.extension_factors, ln_bfs=self.ln_bfs,
                    x_break=self.x_break, show=show)

    def plot_snrs_and_ln_bfs(self, show=False):
        plot_snrs_and_ln_bfs(
            outdir=self.outdir, label=self.label, extension_factors=self.extension_factors, ln_bfs=self.ln_bfs,
            snrs_max_like=self.snrs_max_like,
            snrs_max_like_quantiles=self.snrs_max_like_quantiles, x_break=self.x_break, show=show)

    def plot_delta_bics(self, show=False):
        plot_delta_bics(outdir=self.outdir, label=self.label, extension_factors=self.extension_factors,
                        delta_bics=self.delta_bics, x_break=self.x_break, show=show)

    def plot_log_frequency_spreads(self, show=False):
        plot_log_frequency_spreads(outdir=self.outdir, label=self.label, extension_factors=self.extension_factors,
                                   log_frequency_spreads=self.log_frequency_spreads, x_break=self.x_break, show=show)

    def plot_all(self, show=False):
        self.plot_chi_squares(show)
        self.plot_snrs(show)
        self.plot_ln_bfs(show)
        self.plot_snrs_and_ln_bfs(show)
        self.plot_delta_bics(show)
        self.plot_log_frequency_spreads(show)

    def fill(self, n_snrs=100):
        for start_time, end_time, duration in zip(self.start_times, self.end_times, self.durations):
            label = f"{self.label}_{float(start_time)}_{float(end_time)}"
            self._res_noise = bilby.result.read_in_result(outdir=self.outdir_noise_periodogram, label=label)
            self._res_qpo = bilby.result.read_in_result(outdir=self.outdir_qpo_periodogram, label=label)

            self._calculate_extension_factors(duration=duration)
            self.durations_reduced = np.append(self.durations_reduced, duration)

            self._calculate_idxs(start_time=start_time, end_time=end_time)
            self._calculate_y_selected()
            self._calculate_periodogram()

            self._calculate_ln_bfs()
            self._calculate_log_frequency_spreads()
            self._calculate_delta_bic()
            self._calculate_qpo_max_like_parameters()
            self._calculate_max_like_snr()
            self._calculate_snr_quantiles(n_snrs=n_snrs)
            self._calculate_chi_squares()

            self._calculate_optimal_snr()
            if duration == self.durations[0]:
                self._calculate_max_like_x_break()
            print(self.extension_factors[-1])
        self.snrs_max_like_quantiles = np.array(self.snrs_max_like_quantiles)

    def _calculate_periodogram(self):
        if self.extension_factors[-1] == 1 and self.extension_mode == "zeros":
            window = ("tukey", 0.05)
        else:
            window = "hann"
        self._freqs_combined_periodogram, self._powers_combined_periodogram = \
            periodogram(self._y_selected, fs=self.sampling_frequency, window=window)

    def _calculate_idxs(self, start_time, end_time):
        self._idxs = QPOEstimation.utils.get_indices_by_time(
            times=self.times, minimum_time=start_time, maximum_time=end_time)

    def _calculate_y_selected(self):
        self._y_selected = self.y[self._idxs]
        if self.normalisation:
            self._y_selected = (self._y_selected - np.mean(self._y_selected)) / np.mean(self._y_selected)

    def _calculate_extension_factors(self, duration):
        self.extension_factors = np.append(self.extension_factors, duration / self.durations[0])

    def _calculate_log_frequency_spreads(self):
        self.log_frequency_spreads = \
            np.append(self.log_frequency_spreads, np.std(self._res_qpo.posterior["log_frequency"]))

    def _calculate_ln_bfs(self):
        self.ln_bfs = np.append(self.ln_bfs, self._res_qpo.log_evidence - self._res_noise.log_evidence)

    def _calculate_delta_bic(self):
        bic_qpo = 6 * np.log(len(self._y_selected)) - 2 * self._res_qpo.posterior.iloc[-1]["log_likelihood"]
        bic_noise = 3 * np.log(len(self._y_selected)) - 2 * self._res_noise.posterior.iloc[-1]["log_likelihood"]
        self.delta_bics = np.append(self.delta_bics, bic_qpo - bic_noise)

    def _calculate_chi_squares(self):
        self._calculate_chi_squares_entire_spectrum()
        self._calculate_chi_squares_qpo()
        self._calculate_chi_squares_high_freqs()
        self._calculate_chi_squares_red_noise()

    def _calculate_chi_squares_entire_spectrum(self):
        idxs = QPOEstimation.utils.get_indices_by_time(self._freqs_combined_periodogram,
                                                       minimum_time=1 / self.sampling_frequency - 0.0001,
                                                       maximum_time=self.sampling_frequency / 2)
        self.chi_squares = np.append(self.chi_squares, QPOEstimation.model.psd.periodogram_chi_square_test(
            frequencies=self._freqs_combined_periodogram[idxs], powers=self._powers_combined_periodogram[idxs],
            psd=self._psd_signal_max_like, degrees_of_freedom=len(idxs) - 6))

    def _calculate_chi_squares_qpo(self):
        idxs = QPOEstimation.utils.get_indices_by_time(
            self._freqs_combined_periodogram, minimum_time=self._central_frequency_max_like - 2 * self._width_max_like,
            maximum_time=self._central_frequency_max_like + 2 * self._width_max_like)

        self.chi_squares_qpo = np.append(self.chi_squares_qpo, QPOEstimation.model.psd.periodogram_chi_square_test(
            frequencies=self._freqs_combined_periodogram[idxs], powers=self._powers_combined_periodogram[idxs],
            psd=self._psd_signal_max_like, degrees_of_freedom=len(idxs) - 6))

    def _calculate_chi_squares_red_noise(self):
        try:
            frequency_break_max_like = (self._beta_max_like / self._white_noise_max_like /
                                        self.extension_factors[-1]) ** (1 / self._alpha_max_like)
        except ZeroDivisionError:
            self.chi_squares_red_noise = np.append(self.chi_squares_red_noise, np.nan)
            return
        minimum_frequency = 1/(self.durations[0]/2)
        idxs = QPOEstimation.utils.get_indices_by_time(self._freqs_combined_periodogram,
                                                       minimum_time=minimum_frequency,
                                                       maximum_time=frequency_break_max_like)
        self.chi_squares_red_noise = np.append(
            self.chi_squares_red_noise, QPOEstimation.model.psd.periodogram_chi_square_test(
                frequencies=self._freqs_combined_periodogram[idxs], powers=self._powers_combined_periodogram[idxs],
                psd=self._psd_signal_max_like, degrees_of_freedom=len(idxs) - 6))

    def _calculate_chi_squares_high_freqs(self):
        idxs = QPOEstimation.utils.get_indices_by_time(
            self._freqs_combined_periodogram, minimum_time=self._central_frequency_max_like + self._width_max_like * 2,
            maximum_time=np.max(self._freqs_combined_periodogram))

        self.chi_squares_high_freqs = np.append(
            self.chi_squares_high_freqs, QPOEstimation.model.psd.periodogram_chi_square_test(
                frequencies=self._freqs_combined_periodogram[idxs], powers=self._powers_combined_periodogram[idxs],
                psd=self._psd_signal_max_like, degrees_of_freedom=len(idxs) - 6))

    def _calculate_snr_quantiles(self, n_snrs):
        if n_snrs == 0:
            return None
        snrs = []
        for i in range(n_snrs):
            params = self._res_qpo.posterior.iloc[np.random.randint(0, len(self._res_qpo.posterior))]
            alpha = params.get("alpha", 1)
            beta = np.exp(params.get("log_beta", -30))
            white_noise = np.exp(params["log_sigma"])
            amplitude = np.exp(params["log_amplitude"])
            width = np.exp(params["log_width"])
            central_frequency = np.exp(params["log_frequency"])

            psd_array_noise = QPOEstimation.model.psd.red_noise(
                frequencies=self.frequencies, alpha=alpha,
                beta=beta) + white_noise
            psd_array_qpo = QPOEstimation.model.psd.lorentzian(
                frequencies=self.frequencies, amplitude=amplitude, width=width,
                central_frequency=central_frequency)
            psd_noise = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
                frequency_array=self.frequencies, psd_array=psd_array_noise)
            psd_qpo = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
                frequency_array=self.frequencies, psd_array=psd_array_qpo)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                snr = np.sqrt(np.sum(
                    np.nan_to_num(
                        (psd_qpo.power_spectral_density_interpolated(self._freqs_combined_periodogram) /
                         psd_noise.power_spectral_density_interpolated(self._freqs_combined_periodogram)) ** 2,
                        nan=0)))
            snrs.append(snr)
        # print(np.quantile(snrs, q=[0.05, 0.95]))
        self.snrs_max_like_quantiles.append(np.quantile(snrs, q=[0.05, 0.95]))
        # print(self.snrs_max_like_quantiles)

    def _calculate_qpo_max_like_parameters(self):
        self._alpha_max_like = self._res_qpo.posterior.iloc[-1].get("alpha", 0)
        self._beta_max_like = np.exp(self._res_qpo.posterior.iloc[-1].get("log_beta", -100))
        self._white_noise_max_like = np.exp(self._res_qpo.posterior.iloc[-1].get("log_sigma", -100))
        self._amplitude_max_like = np.exp(self._res_qpo.posterior.iloc[-1].get("log_amplitude", -100))
        self._width_max_like = np.exp(self._res_qpo.posterior.iloc[-1].get("log_width", -100))
        self._central_frequency_max_like = np.exp(self._res_qpo.posterior.iloc[-1].get("log_frequency", -100))
        self.max_like_parameters = dict(alpha=self._alpha_max_like,
                                        beta=self._beta_max_like,
                                        white_nosie=self._white_noise_max_like,
                                        amplitude=self._amplitude_max_like,
                                        width=self._width_max_like,
                                        central_frequency=self._central_frequency_max_like)

    def _calculate_max_like_snr(self):
        self._psd_array_noise_max_like = QPOEstimation.model.psd.red_noise(
            frequencies=self.frequencies, alpha=self._alpha_max_like,
            beta=self._beta_max_like) + self._white_noise_max_like
        self._psd_array_qpo_max_like = QPOEstimation.model.psd.lorentzian(
            frequencies=self.frequencies, amplitude=self._amplitude_max_like, width=self._width_max_like,
            central_frequency=self._central_frequency_max_like)
        self._psd_array_signal_max_like = self._psd_array_noise_max_like + self._psd_array_qpo_max_like

        self._psd_noise_max_like = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=self.frequencies, psd_array=self._psd_array_noise_max_like)
        self._psd_qpo_max_like = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=self.frequencies, psd_array=self._psd_array_qpo_max_like)
        self._psd_signal_max_like = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=self.frequencies, psd_array=self._psd_array_signal_max_like)

        snr_max_like = self._calculate_snr(freqs=self._freqs_combined_periodogram, psd_signal=self._psd_qpo_max_like,
                                           psd_noise=self._psd_noise_max_like)
        self.snrs_max_like = np.append(self.snrs_max_like, snr_max_like)

    def _calculate_optimal_snr(self):
        if None in [self.injection_parameters, self.injection_psds, self.extension_mode]:
            return
        # freqs_combined_periodogram, powers_combined_periodogram = \
        #     periodogram(self._y_selected, fs=self.sampling_frequency, window="hann")

        if self.extension_mode == "zeros":
            psd_array_noise_diluted = \
                self.injection_psds["red_noise"].psd_array / self.extension_factors[-1] \
                + self.injection_parameters["sigma"] / self.extension_factors[-1]
        else:
            psd_array_noise_diluted = \
                self.injection_psds["red_noise"].psd_array / self.extension_factors[-1] \
                + self.injection_parameters["sigma"]

        psd_array_qpo_diluted = self.injection_psds["qpo"].psd_array / self.extension_factors[-1]

        psd_noise_diluted = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=self.frequencies, psd_array=psd_array_noise_diluted)
        psd_qpo_diluted = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=self.frequencies, psd_array=psd_array_qpo_diluted)
        snr_optimal = self._calculate_snr(freqs=self._freqs_combined_periodogram,
                                          psd_signal=psd_qpo_diluted, psd_noise=psd_noise_diluted)
        self.snrs_optimal = np.append(self.snrs_optimal, snr_optimal)

    @staticmethod
    def _calculate_snr(freqs, psd_signal, psd_noise):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal = psd_signal.power_spectral_density_interpolated(freqs)
            noise = psd_noise.power_spectral_density_interpolated(freqs)
            snr_squared = (signal / noise) ** 2
            return np.sqrt(np.sum(np.nan_to_num(snr_squared, nan=0)))
