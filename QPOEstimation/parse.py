import argparse


LIKELIHOOD_MODELS = ["celerite", "celerite_windowed", "george", "whittle"]
MODES = ["qpo", "white_noise", "red_noise", "pure_qpo", "qpo_plus_red_noise",
         "double_red_noise", "double_qpo", "matern32", "broken_power_law", "matern52", "exp_sine2",
         "rational_quadratic",  "exp_squared", "exp_sine2_rn"]
DATA_SOURCES = ["injection", "giant_flare", "solar_flare", "grb", "magnetar_flare",
                "magnetar_flare_binned", "hares_and_hounds"]
RUN_MODES = ["select_time", "sliding_window", "entire_segment", "from_maximum"]
BACKGROUND_MODELS = ["polynomial", "exponential", "skew_exponential", "gaussian", "log_normal", "lorentzian", "mean",
                     "skew_gaussian", "fred", "fred_extended", "0"]
GRB_ENERGY_BANDS = ["15-25", "25-50", "50-100", "100-350", "15-350", "all"]
OSCILLATORY_MODELS = ["qpo", "pure_qpo", "qpo_plus_red_noise", "fourier_series"]


def parse_args() -> argparse.ArgumentParser:
    """ Creates an argument parser for the analysis scripts.

    Returns
    -------
    The argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="giant_flare", choices=DATA_SOURCES)
    parser.add_argument("--run_mode", default="sliding_window", choices=RUN_MODES)
    parser.add_argument("--sampling_frequency", default=None, type=int)
    parser.add_argument("--variance_stabilisation", default="False", type=str)

    parser.add_argument("--hares_and_hounds_id", default="5700", type=str)
    parser.add_argument("--hares_and_hounds_round", default="HH2", type=str)

    parser.add_argument("--solar_flare_folder", default="goes", type=str)
    parser.add_argument("--solar_flare_id", default="120704187", type=str)
    parser.add_argument("--grb_id", default="090709A", type=str)
    parser.add_argument("--grb_label", default="ASIM_CLEANED_LED", type=str)
    parser.add_argument("--grb_binning", default="1s", type=str)
    parser.add_argument("--grb_detector", default="swift", type=str)
    parser.add_argument("--grb_energy_band", default="all", choices=GRB_ENERGY_BANDS, type=str)

    parser.add_argument("--magnetar_label", default="SGR_1806_20", type=str)
    parser.add_argument("--magnetar_tag", default="10223-01-03-010_90907122.0225", type=str)
    parser.add_argument("--bin_size", default=0.001, type=float)
    parser.add_argument("--magnetar_subtract_t0", default="True", type=str)
    parser.add_argument("--magnetar_unbarycentred_time", default="False", type=str)
    parser.add_argument("--rebin_factor", default=1, type=int)

    parser.add_argument("--start_time", default=0., type=float)
    parser.add_argument("--end_time", default=1., type=float)

    parser.add_argument("--period_number", default=0, type=int)
    parser.add_argument("--run_id", default=0, type=int)

    parser.add_argument("--candidate_id", default=0, type=int)

    parser.add_argument("--injection_id", default=0, type=int)
    parser.add_argument("--injection_file_dir", default="injection_files", type=str)
    parser.add_argument("--injection_mode", default="qpo", choices=MODES, type=str)
    parser.add_argument("--injection_likelihood_model", default="qpo_plus_red_noise", choices=LIKELIHOOD_MODELS, type=str)
    parser.add_argument("--base_injection_outdir", default="injections/injection", type=str)

    parser.add_argument("--offset", default="False", type=str)
    parser.add_argument("--polynomial_max", default=1000, type=float)
    parser.add_argument("--amplitude_min", default=None, type=float)
    parser.add_argument("--amplitude_max", default=None, type=float)
    parser.add_argument("--offset_min", default=None, type=float)
    parser.add_argument("--offset_max", default=None, type=float)
    parser.add_argument("--sigma_min", default=None, type=float)
    parser.add_argument("--sigma_max", default=None, type=float)
    parser.add_argument("--t_0_min", default=None, type=float)
    parser.add_argument("--t_0_max", default=None, type=float)

    parser.add_argument("--min_log_a", default=None, type=float)
    parser.add_argument("--max_log_a", default=None, type=float)
    parser.add_argument("--min_log_c_red_noise", default=None, type=float)
    parser.add_argument("--min_log_c_qpo", default=None, type=float)
    parser.add_argument("--max_log_c_red_noise", default=None, type=float)
    parser.add_argument("--max_log_c_qpo", default=None, type=float)
    parser.add_argument("--minimum_window_spacing", default=0, type=float)

    parser.add_argument("--recovery_mode", default="qpo", choices=MODES)
    parser.add_argument("--likelihood_model", default="celerite", choices=LIKELIHOOD_MODELS)
    parser.add_argument("--normalisation", default="False", type=str)
    parser.add_argument("--background_model", default="polynomial", choices=BACKGROUND_MODELS)
    parser.add_argument("--n_components", default=1, type=int)
    parser.add_argument("--jitter_term", default="False", type=str)
    parser.add_argument("--window", default="hann", choices=["hann", "tukey", "boxcar"])
    parser.add_argument("--frequency_mask_minimum", default=None, type=float)
    parser.add_argument("--frequency_mask_maximum", default=None, type=float)

    parser.add_argument("--band_minimum", default=None, type=float)
    parser.add_argument("--band_maximum", default=None, type=float)

    parser.add_argument("--segment_length", default=1.0, type=float)
    parser.add_argument("--segment_step", default=0.27, type=float)

    parser.add_argument("--nlive", default=150, type=int)
    parser.add_argument("--sample", default="rwalk", type=str)
    parser.add_argument("--use_ratio", default="False", type=str)

    parser.add_argument("--try_load", default="True", type=str)
    parser.add_argument("--resume", default="False", type=str)
    parser.add_argument("--plot", default="True", type=str)
    parser.add_argument("--suffix", default="", type=str)
    return parser
