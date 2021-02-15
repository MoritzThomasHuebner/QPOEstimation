import bilby
import json
import numpy as np
from copy import deepcopy
from QPOEstimation.prior.minimum import MinimumPrior

import sys
import argparse


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum_id", default=0, type=int)
    parser.add_argument("--maximum_id", default=100, type=int)
    parser.add_argument("--injection_mode", default="qpo", choices=["qpo", "white_noise", "red_noise", "general_qpo", "pure_qpo"], type=str)
    parser.add_argument("--likelihood_model", default="gaussian_process",
                        choices=["gaussian_process", "gaussian_process_windowed"], type=str)
    args = parser.parse_args()
    minimum_id = args.minimum_id
    maximum_id = args.maximum_id
    injection_mode = args.injection_mode
    likelihood_model = args.likelihood_model
else:
    minimum_id = 0
    maximum_id = 100

    injection_mode = "qpo"
    likelihood_model = "gaussian_process"

samples = []

band_minimum = 5
band_maximum = 64

reslist = []
for i in range(minimum_id, maximum_id):
    try:
        with open(f'injection_files/{injection_mode}/{str(i).zfill(2)}_params.json') as f:
            injection_params = json.load(f)
        res = bilby.result.read_in_result(f'injection_{band_minimum}_{band_maximum}Hz_normal_{injection_mode}/{injection_mode}/{likelihood_model}/results/{str(i).zfill(2)}_{likelihood_model}_result.json')
        reslist.append(res)
        reslist[-1].injection_parameters = injection_params
    except (OSError, FileNotFoundError) as e:
        print(e)
        continue

if likelihood_model == 'gaussian_process_windowed':
    bilby.result.make_pp_plot(results=reslist, filename=f'{injection_mode}_pp_plot_windowed_new.png')
else:
    bilby.result.make_pp_plot(results=reslist, filename=f'{injection_mode}_pp_plot_unwindowed_new.png')
