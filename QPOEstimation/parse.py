import argparse

likelihood_models = ["gaussian_process", "gaussian_process_windowed", "george_likelihood", "whittle"]
modes = ["qpo", "white_noise", "red_noise", "pure_qpo", "general_qpo",
         "double_red_noise", "double_qpo", 'matern32', "broken_power_law", "matern52", "exp_sine2",
         "rational_quadratic",  "exp_squared", "exp_sine2_rn"]
data_sources = ['injection', 'giant_flare', 'solar_flare', 'grb', 'magnetar_flare',
                'magnetar_flare_binned', 'hares_and_hounds']
run_modes = ['select_time', 'sliding_window', 'candidates', 'entire_segment', 'from_maximum']
background_models = ["polynomial", "exponential", "skew_exponential", "gaussian", "log_normal", "lorentzian", "mean",
                     "skew_gaussian", "fred", "fred_extended", "0"]
data_modes = ['normal', 'smoothed', 'smoothed_residual']
grb_energy_bands = ['15-25', '25-50', '50-100', '100-350', '15-350', 'all']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default='giant_flare', choices=data_sources)
    parser.add_argument("--run_mode", default='sliding_window', choices=run_modes)
    parser.add_argument("--sampling_frequency", default=None, type=int)
    parser.add_argument("--data_mode", choices=data_modes, default='normal', type=str)
    parser.add_argument("--alpha", default=0.02, type=float)
    parser.add_argument("--variance_stabilisation", default='False', type=str)

    parser.add_argument("--hares_and_hounds_id", default='5700', type=str)
    parser.add_argument("--hares_and_hounds_round", default='HH2', type=str)

    parser.add_argument("--solar_flare_folder", default='goes', type=str)
    parser.add_argument("--solar_flare_id", default='120704187', type=str)
    parser.add_argument("--grb_id", default='090709A', type=str)
    parser.add_argument("--grb_label", default='ASIM_CLEANED_LED', type=str)
    parser.add_argument("--grb_binning", default='1s', type=str)
    parser.add_argument("--grb_detector", default='swift', type=str)
    parser.add_argument("--grb_energy_band", default='all', choices=grb_energy_bands, type=str)

    parser.add_argument("--magnetar_label", default='SGR_1806_20', type=str)
    parser.add_argument("--magnetar_tag", default='10223-01-03-010_90907122.0225', type=str)
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
    parser.add_argument("--injection_mode", default="qpo", choices=modes, type=str)
    parser.add_argument("--injection_likelihood_model", default="general_qpo", choices=likelihood_models, type=str)
    parser.add_argument("--base_injection_outdir", default="injections/injection", type=str)

    parser.add_argument("--offset", default='False', type=str)
    parser.add_argument("--polynomial_max", default=1000, type=float)
    parser.add_argument("--amplitude_min", default=None, type=float)
    parser.add_argument("--amplitude_max", default=None, type=float)
    parser.add_argument("--offset_min", default=None, type=float)
    parser.add_argument("--offset_max", default=None, type=float)
    parser.add_argument("--sigma_min", default=None, type=float)
    parser.add_argument("--sigma_max", default=None, type=float)
    parser.add_argument("--tau_min", default=None, type=float)
    parser.add_argument("--tau_max", default=None, type=float)
    parser.add_argument("--t_0_min", default=None, type=float)
    parser.add_argument("--t_0_max", default=None, type=float)

    parser.add_argument("--min_log_a", default=None, type=float)
    parser.add_argument("--max_log_a", default=None, type=float)
    parser.add_argument("--min_log_c_red_noise", default=None, type=float)
    parser.add_argument("--min_log_c_qpo", default=None, type=float)
    parser.add_argument("--max_log_c_red_noise", default=None, type=float)
    parser.add_argument("--max_log_c_qpo", default=None, type=float)
    parser.add_argument("--minimum_window_spacing", default=0, type=float)

    parser.add_argument("--recovery_mode", default="qpo", choices=modes)
    parser.add_argument("--likelihood_model", default="gaussian_process", choices=likelihood_models)
    parser.add_argument("--normalisation", default="False", type=str)
    parser.add_argument("--background_model", default="polynomial", choices=background_models)
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
    parser.add_argument("--sample", default='rwalk', type=str)
    parser.add_argument("--use_ratio", default='False', type=str)

    parser.add_argument("--try_load", default='True', type=str)
    parser.add_argument("--resume", default='False', type=str)
    parser.add_argument("--plot", default='True', type=str)
    parser.add_argument("--suffix", default='', type=str)
    return parser
