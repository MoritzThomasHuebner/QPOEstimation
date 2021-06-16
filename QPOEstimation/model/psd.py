import numpy as np


def red_noise(frequencies, alpha, beta):
    return beta * frequencies ** (-alpha)


def white_noise(frequencies, sigma):
    return sigma * np.ones(len(frequencies))


def broken_power_law_noise(frequencies, alpha_1, alpha_2, beta, delta, rho):
    return beta * frequencies ** (-alpha_1) * (1 + (frequencies / delta) ** ((alpha_2 - alpha_1) / rho)) ** (-rho)


def lorentzian(frequencies, amplitude, central_frequency, width):
    return amplitude * (width ** 2 / ((frequencies - central_frequency) ** 2 + width ** 2)) / np.pi / width


def periodogram_chi_square_test(frequencies, powers, psd, degrees_of_freedom):
    # Nita et al 2014
    return np.sum((1 - powers/psd.power_spectral_density_interpolated(frequencies))**2)/degrees_of_freedom
