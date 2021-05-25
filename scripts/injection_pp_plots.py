import argparse
import json
import sys

import bilby

from QPOEstimation.result import GPResult
from QPOEstimation.utils import get_injection_outdir, modes, likelihood_models
import matplotlib.pyplot as plt
plt.style.use("paper.mplstyle")

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum_id", default=0, type=int)
    parser.add_argument("--maximum_id", default=100, type=int)
    parser.add_argument("--injection_mode", default="qpo", choices=modes, type=str)
    parser.add_argument("--likelihood_model", default="gaussian_process",
                        choices=likelihood_models, type=str)
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

reslist = []
outdir = get_injection_outdir(injection_mode=injection_mode, recovery_mode=injection_mode,
                              likelihood_model=likelihood_model)
outdir = f"{outdir}/results"

for i in range(minimum_id, maximum_id):
    try:
        with open(f'injection_files/{injection_mode}/{likelihood_model}/{str(i).zfill(2)}_params.json') as f:
            injection_params = json.load(f)
        label = f"{str(i).zfill(2)}"
        res = GPResult.from_json(outdir=outdir, label=label)
        reslist.append(res)
        reslist[-1].injection_parameters = injection_params
    except (OSError, FileNotFoundError) as e:
        print(e)
        continue

bilby.result.make_pp_plot(results=reslist, filename=f'{injection_mode}_{likelihood_model}_pp_plot.png')
