import json
import bilby
import matplotlib.pyplot as plt
from pathlib import Path

injection_mode = "general_qpo"
band_minimum = 5
band_maximum = 64

Path(f'{injection_mode}_injections/').mkdir(parents=True, exist_ok=True)

for i in range(100, 200):
    try:
        with open(f'injection_files/{injection_mode}/{str(i).zfill(2)}_params.json') as f:
            injection_params = json.load(f)
        res = bilby.result.read_in_result(f'injection_{band_minimum}_{band_maximum}Hz_normal_{injection_mode}/{injection_mode}/results/{str(i).zfill(2)}_gaussian_process_windowed_result.json')
        plt.plot(res.posterior['window_maximum'])
        plt.axhline(injection_params['window_maximum'])
        plt.savefig(f'{injection_mode}_injections/{str(i).zfill(2)}.png')
        plt.clf()
    except (OSError, FileNotFoundError) as e:
        print(e)
        continue
