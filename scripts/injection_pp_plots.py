import argparse
import json
import sys

import bilby

from QPOEstimation.result import GPResult
from QPOEstimation.utils import get_injection_outdir, get_injection_label
from QPOEstimation.parse import LIKELIHOOD_MODELS, MODES
import matplotlib.pyplot as plt

plt.style.use("paper.mplstyle")

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum_id", default=0, type=int)
    parser.add_argument("--maximum_id", default=100, type=int)
    parser.add_argument("--injection_mode", default="qpo", choices=MODES, type=str)
    parser.add_argument("--likelihood_model", default="celerite",
                        choices=LIKELIHOOD_MODELS, type=str)
    parser.add_argument("--base_injection_outdir", default="injection", type=str)
    args = parser.parse_args()
    minimum_id = args.minimum_id
    maximum_id = args.maximum_id
    injection_mode = args.injection_mode
    likelihood_model = args.likelihood_model
    base_injection_outdir = args.base_injection_outdir
else:
    minimum_id = 0
    maximum_id = 100

    injection_mode = "qpo_plus_red_noise"
    likelihood_model = "celerite"
    base_injection_outdir = "injections/injection_pp_non_eq_dis"

samples = []
reslist = []
outdir = get_injection_outdir(injection_mode=injection_mode, recovery_mode=injection_mode,
                              likelihood_model=likelihood_model, base_injection_outdir=base_injection_outdir)
outdir = f"{outdir}/results"

for injection_id in range(minimum_id, maximum_id):
    print(injection_id)
    label = get_injection_label(run_mode="entire_segment", injection_id=injection_id) + "_1_skew_gaussians"
    try:
        with open(f"{base_injection_outdir}/{injection_mode}/{likelihood_model}/{str(injection_id).zfill(2)}_params.json") as f:
            injection_params = json.load(f)
        res = GPResult.from_json(outdir=outdir, label=label)
        reslist.append(res)
        reslist[-1].injection_parameters = injection_params
    except (OSError, FileNotFoundError) as e:
        print(e)
        continue

for key in reslist[0].priors:
    reslist[0].priors[key].latex_label = reslist[0].priors[key].latex_label.replace("_", "\\_")

reslist[0].priors["kernel:terms[0]:log_a"].latex_label = "$a_{\mathrm{qpo}}$"
reslist[0].priors["kernel:terms[0]:log_c"].latex_label = "$c_{\mathrm{qpo}}$"
reslist[0].priors["kernel:terms[0]:log_f"].latex_label = "$f_{\mathrm{qpo}}$"

reslist[0].priors["kernel:terms[1]:log_a"].latex_label = "$a_{\mathrm{rn}}$"
reslist[0].priors["kernel:terms[1]:log_c"].latex_label = "$c_{\mathrm{rn}}$"

reslist[0].priors["mean:log_amplitude_0"].latex_label = "$A$"
reslist[0].priors["mean:log_sigma_rise_0"].latex_label = "$\sigma_1$"
reslist[0].priors["mean:log_sigma_fall_0"].latex_label = "$\sigma_2$"
reslist[0].priors["mean:t_0_0"].latex_label = "$t_0$"


bilby.result.make_pp_plot(results=reslist, filename=f"results/{injection_mode}_{likelihood_model}_pp_plot_non_eq_dis.pdf")
