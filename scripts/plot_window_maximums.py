import json
import bilby
import matplotlib.pyplot as plt
from pathlib import Path

from QPOEstimation.utils import get_injection_outdir

injection_mode = "qpo_plus_red_noise"
band_minimum = 5
band_maximum = 64

band = f"{band_minimum}_{band_maximum}Hz"

Path(f"injections/{injection_mode}_injections/").mkdir(parents=True, exist_ok=True)

for i in range(100, 200):
    try:
        with open(f"injections/injection_files/{injection_mode}/{str(i).zfill(2)}_params.json") as f:
            injection_params = json.load(f)

        res = bilby.result.read_in_result(outdir=get_injection_outdir(
            injection_mode=injection_mode, recovery_mode=injection_mode,
            likelihood_model="celerite_windowed"), label=f"{str(i).zfill(2)}")
        plt.hist(res.posterior["window_maximum"], bins="fd", density=True)
        plt.axvline(injection_params["window_maximum"], color="orange")
        plt.savefig(f"injections/{injection_mode}_injections/{str(i).zfill(2)}.png")
        plt.clf()
    except (OSError, FileNotFoundError) as e:
        print(e)
        continue
