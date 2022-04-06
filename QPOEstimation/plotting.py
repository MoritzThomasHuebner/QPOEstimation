import matplotlib.pyplot as plt
import numpy as np

"""
This module contains a number of plotting utilities. Partly for the analysis of the 'Pitfalls of Periodograms' paper,
in part plotting utilities for Gaussian processes.
"""


def plot_chi_squares(
        outdir: str, label: str, extension_factors: np.ndarray, chi_squares: np.ndarray = None,
        chi_squares_qpo: np.ndarray = None, chi_squares_red_noise: np.ndarray = None,
        chi_squares_high_freqs: np.ndarray = None, show: bool = False) -> None:
    """ Plots the chi-square values against the extension factors. """
    if chi_squares is not None:
        plt.plot(extension_factors, chi_squares, label="Entire spectrum")
    if chi_squares_qpo is not None:
        plt.plot(extension_factors, chi_squares_qpo, label="$f_0 \pm 2\sigma$")
    # if chi_squares_red_noise is not None:
    #     if not np.isnan(chi_squares_red_noise[0]):
    #         plt.plot(extension_factors, chi_squares_red_noise, label="$f \leq f_{\mathrm{break}, x}$")
    if chi_squares_high_freqs is not None:
        plt.plot(extension_factors, chi_squares_high_freqs, label="$f \geq f_0 + 2\sigma$")
    plt.xlabel("$x$")
    plt.ylabel("$\chi^2$")
    top = np.max(np.nan_to_num(
        np.array([chi_squares, chi_squares_qpo, chi_squares_red_noise, chi_squares_high_freqs]), nan=0))
    top = min(1.5, top) + 0.1
    print(top)
    # plt.ylim(0, top=top)
    plt.legend(ncol=2)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05),
               fancybox=True, shadow=False, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{label}_chi_squares_vs_extension.pdf")
    if show:
        plt.show()
    plt.clf()


def plot_snrs(
        outdir: str, label: str, extension_factors: np.ndarray, snrs_optimal: np.ndarray = None,
        snrs_max_like: np.ndarray = None, snrs_max_like_quantiles: np.ndarray = None, x_break: float = None,
        show: bool = False) -> None:
    """ Plots the SNR values against the extension factors. """
    if len(snrs_optimal) != 0:
        plt.plot(extension_factors, snrs_optimal, label="Optimal SNR")
    if snrs_max_like is not None:
        plt.plot(extension_factors, snrs_max_like, label="Maximum likelihood SNR")
    if len(snrs_max_like_quantiles) != 0:
        plt.fill_between(extension_factors, snrs_max_like_quantiles[:, 0], snrs_max_like_quantiles[:, 1], color="#ff7f0e", alpha=0.3)
    if x_break is not None:
        plt.axvline(x_break, color="black", linestyle="-.", label="$x_{\mathrm{break}}$")
    plt.xlabel("$x$")
    plt.ylabel("SNR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/{label}_snr_vs_extension.pdf")
    if show:
        plt.show()
    plt.clf()


def plot_ln_bfs(
        outdir: str, label: str, extension_factors: np.ndarray, ln_bfs: np.ndarray,
        x_break: float = None, show: bool = False) -> None:
    """ Plots the ln BF values against the extension factors. """
    plt.plot(extension_factors, ln_bfs)
    if x_break is not None:
        plt.axvline(x_break, color="red", linestyle="-.", label="$x_{\mathrm{break}}$")
    plt.xlabel("$x$")
    plt.ylabel("$\ln BF$")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{label}_ln_bf_vs_extension.pdf")
    if show:
        plt.show()
    plt.clf()


def plot_snrs_and_ln_bfs(
        outdir: str, label: str, extension_factors: np.ndarray, ln_bfs: np.ndarray, snrs_max_like: np.ndarray = None,
        snrs_max_like_quantiles: np.ndarray = None, x_break: float = None, show: bool = False) -> None:
    """ Plots the SNR and ln BF values against the extension factors. """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots()
    # if len(snrs_optimal) != 0:
    #     lines.append(ax.plot(extension_factors, snrs_optimal, label="Optimal SNR"))
    # if snrs_max_like is not None:
    ax.plot(extension_factors, snrs_max_like, color=colors[0])

    if len(snrs_max_like_quantiles) != 0:
        ax.fill_between(extension_factors, snrs_max_like_quantiles[:, 0], snrs_max_like_quantiles[:, 1], color=colors[0], alpha=0.3)
    if x_break is not None:
        ax.axvline(x_break, color="black", linestyle="-.", label="$x_{\mathrm{break}}$")
        ax.legend()
    # if True:
    #     ax.axvline(3, color="black", linestyle="-.", label="$x=3$")
    #     ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("SNR", color=colors[0])
    ax.tick_params(axis="y", labelcolor=colors[0])

    ax2 = ax.twinx()
    ax2.set_ylabel("$\ln BF$", color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])
    ax2.plot(extension_factors, ln_bfs, color=colors[1])

    fig.tight_layout()
    plt.xlim(extension_factors[0], extension_factors[-1])
    fig.savefig(f"{outdir}/{label}_snrs_and_ln_bf_vs_extension.pdf")
    if show:
        fig.show()
    plt.clf()


def plot_delta_bics(
        outdir: str, label: str, extension_factors: np.ndarray, delta_bics: np.ndarray, x_break: float = None,
        show: bool = False) -> None:
    """ Plots the Delta BIC values against the extension factors. """
    plt.plot(extension_factors, delta_bics, label="Inferred $\Delta BIC$")
    if x_break is not None:
        plt.axvline(x_break, color="red", linestyle="-.", label="$x_{\mathrm{break}}$")
    plt.xlabel("$x$")
    plt.ylabel("$\Delta BIC$")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{label}_delta_bic_vs_extension_factor.pdf")
    if show:
        plt.show()
    plt.clf()


def plot_log_frequency_spreads(
        outdir: str, label: str, extension_factors: np.ndarray, log_frequency_spreads: np.ndarray,
        x_break: float = None, show: bool = False) -> None:
    """ Plots the log frequency spread values against the extension factors. """
    plt.plot(extension_factors, log_frequency_spreads)
    plt.xlabel("$x$")
    plt.ylabel("$\Delta (\ln f)$")
    if x_break is not None:
        plt.axvline(x_break, color="black", linestyle="-.", label="$x_{\mathrm{break}}$")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{label}_ln_f_spread_vs_extension_factor.pdf")
    if show:
        plt.show()
    plt.clf()
