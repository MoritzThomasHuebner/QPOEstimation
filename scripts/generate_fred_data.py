import bilby
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
import numpy as np

from QPOEstimation.model.celerite import get_n_component_fred_model
from QPOEstimation.likelihood import get_kernel
from QPOEstimation.injection import InjectionCreator
import json

with open('SolarFlare/120704187/results/fred_1_result.json') as f:
    ref_res = json.load(f)

# ref_res = bilby.result.read_in_result('SolarFlare/120704187/results/initial_test_fred_3_result.json')
max_like_params = ref_res['posterior']['content']
for key, value in max_like_params.items():
    max_like_params[key] = value[-1]
max_like_mean_params = dict()
max_like_kernel_params = dict()
mean_model = get_n_component_fred_model(n_freds=3)
kernel = get_kernel(kernel_type='qpo')
times = np.arange(350, 1400)
creator = InjectionCreator(params=max_like_params, injection_mode='qpo', times=times,
                           outdir='solar_flare_injection', injection_id='1_fred_injection', likelihood_model='gaussian_process', mean_model=mean_model,
                           poisson_data=True)

plt.plot(times, creator.mean_model.get_value(times))
plt.show()


plt.plot(times, creator.mean_model.get_value(times), label='mean')
plt.errorbar(times, creator.y, yerr=creator.yerr, fmt=".k", capsize=0, label='Generated GP data')
plt.ylabel('Photon counts')
plt.xlabel('time[s]')
plt.legend()
plt.tight_layout()
plt.show()

creator.save()

# for key, value in max_like_params.items():
#     if key.startswith('mean:'):
#         mean_model.set_parameter(key.replace('mean:', ''), value[-1])
#     elif key.startswith('kernel:'):
#         kernel.set_parameter(key.replace('kernel:', ''), value[-1])
#
# counts = mean_model(times)
