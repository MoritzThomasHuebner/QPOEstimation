import bilby
import numpy as np


def inject_model_into_noise(frequencies, psd_array, model):
    psd = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_array(
        frequency_array=frequencies,
        psd_array=psd_array)
    series = bilby.core.series.CoupledTimeAndFrequencySeries()
    series.frequency_array = frequencies
    asd_realisation, frequencies = psd.get_noise_realisation(sampling_frequency=series.sampling_frequency,
                                                             duration=series.duration)
    norm = 0.5 * np.sqrt(series.duration)

    asd_realisation = np.real(asd_realisation) / norm
    asd_realisation += np.sqrt(model)
    return asd_realisation
