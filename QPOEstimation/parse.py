from utils import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default='giant_flare', choices=data_sources)
    parser.add_argument("--run_mode", default='sliding_window', choices=run_modes)
    parser.add_argument("--sampling_frequency", default=None, type=int)
    parser.add_argument("--data_mode", choices=data_modes, default='normal', type=str)
    parser.add_argument("--alpha", default=0.02, type=float)
    parser.add_argument("--variance_stabilisation", default='True', type=str)

    parser.add_argument("--solar_flare_id", default='120704187', type=str)

    parser.add_argument("--start_time", default=0., type=float)
    parser.add_argument("--end_time", default=1., type=float)

    parser.add_argument("--period_number", default=0, type=int)
    parser.add_argument("--run_id", default=0, type=int)

    parser.add_argument("--candidate_id", default=0, type=int)

    parser.add_argument("--injection_id", default=0, type=int)
    parser.add_argument("--injection_mode", default="qpo", choices=modes, type=str)

    parser.add_argument("--polynomial_max", default=1000, type=float)
    parser.add_argument("--min_log_a", default=-5, type=float)
    parser.add_argument("--max_log_a", default=15, type=float)
    parser.add_argument("--min_log_c", default=-6, type=float)
    parser.add_argument("--max_log_c", default=1, type=float)
    parser.add_argument("--minimum_window_spacing", default=0, type=float)

    parser.add_argument("--recovery_mode", default="qpo", choices=modes)
    parser.add_argument("--model", default="gaussian_process", choices=likelihood_models)
    parser.add_argument("--background_model", default="polynomial", choices=background_models)
    parser.add_argument("--n_components", default=1, type=int)

    parser.add_argument("--band_minimum", default=10, type=int)
    parser.add_argument("--band_maximum", default=32, type=int)

    parser.add_argument("--segment_length", default=1.0, type=float)
    parser.add_argument("--segment_step", default=0.27, type=float)

    parser.add_argument("--nlive", default=150, type=int)
    parser.add_argument("--sample", default='rwalk', type=str)
    parser.add_argument("--use_ratio", default='False', type=str)

    parser.add_argument("--try_load", default='True', type=str)
    parser.add_argument("--resume", default='False', type=str)
    parser.add_argument("--plot", default='True', type=str)
    return parser
