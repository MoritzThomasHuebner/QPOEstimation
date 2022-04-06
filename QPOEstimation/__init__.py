from . import likelihood, prior, poisson, smoothing, model, stabilisation, \
    injection, get_data, post_processing, plotting, result, utils

LIKELIHOOD_MODELS = ["celerite", "celerite_windowed", "george", "whittle"]
MODES = ["qpo", "white_noise", "red_noise", "pure_qpo", "qpo_plus_red_noise",
         "double_red_noise", "double_qpo", "matern32", "broken_power_law", "matern52", "exp_sine2",
         "rational_quadratic",  "exp_squared", "exp_sine2_rn"]
DATA_SOURCES = ["injection", "giant_flare", "solar_flare", "grb", "magnetar_flare",
                "magnetar_flare_binned", "hares_and_hounds"]
RUN_MODES = ["select_time", "sliding_window", "entire_segment", "from_maximum"]
BACKGROUND_MODELS = ["polynomial", "exponential", "skew_exponential", "gaussian", "log_normal", "lorentzian", "mean",
                     "skew_gaussian", "fred", "fred_extended", "0"]
DATA_MODES = ["normal", "smoothed", "smoothed_residual"]
GRB_ENERGY_BANDS = ["15-25", "25-50", "50-100", "100-350", "15-350", "all"]
OSCILLATORY_MODELS = ["qpo", "pure_qpo", "qpo_plus_red_noise", "fourier_series"]