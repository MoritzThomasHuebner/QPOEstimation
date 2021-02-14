import bilby
import json
import numpy as np
from copy import deepcopy
from QPOEstimation.prior.minimum import MinimumPrior

samples = []
injection_mode = 'red_noise'
polynomial_max = 0
min_log_a = -2
max_log_a = 1
min_log_c = -1

band_minimum = 5
band_maximum = 64

segment_length = 1
sampling_frequency = 256
t = np.linspace(0, segment_length, int(sampling_frequency * segment_length))

# for likelihood_model in ['gaussian_process', 'gaussian_process_windowed']:
for likelihood_model in ['gaussian_process']:
    reslist = []
    for i in range(0, 100):
        try:
            with open(f'injection_files/{injection_mode}/{str(i).zfill(2)}_params.json') as f:
                injection_params = json.load(f)

            if likelihood_model == 'gaussian_process_windowed':
                res = bilby.result.read_in_result(f'injection_{band_minimum}_{band_maximum}Hz_normal_{injection_mode}/{injection_mode}/results/{str(i).zfill(2)}_gaussian_process_windowed_result.json')
            else:
                res = bilby.result.read_in_result(f'injection_{band_minimum}_{band_maximum}Hz_normal_{injection_mode}/{injection_mode}/results/{str(i).zfill(2)}_gaussian_process_result.json')

            reslist.append(res)
            reslist[-1].injection_parameters = injection_params
        except (OSError, FileNotFoundError) as e:
            print(e)
            continue


    if likelihood_model == 'gaussian_process_windowed':
        bilby.result.make_pp_plot(results=reslist, filename=f'{injection_mode}_pp_plot_windowed_new.png')
    else:
        bilby.result.make_pp_plot(results=reslist, filename=f'{injection_mode}_pp_plot_unwindowed_new.png')