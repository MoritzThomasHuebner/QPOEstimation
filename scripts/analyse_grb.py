import numpy as np
import bilby
import QPOEstimation
from QPOEstimation.result import GPResult
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')

evidence_dict = dict()
evidence_err_dict = dict()

mean_model = 'fred_norris'

for mean_model in ['fred', 'fred_norris', 'fred_norris_extended', 'skew_gaussian']:
    for recovery_mode in ['general_qpo', 'red_noise']:
        evidences = []
        evidence_errs = []
        for n_components in range(5):
            try:
                res = GPResult.from_json(
                    outdir=f'GRB090709A_swift/select_time/{recovery_mode}/'
                           f'gaussian_process/results/',
                    label=f'-4.0_103.0_{n_components}_{mean_model}s')
                evidences.append(res.log_evidence)
                evidence_errs.append(res.log_evidence_err)
                print(f'{recovery_mode}\t{n_components}\t{res.log_evidence}')
            except Exception as e:
                print(e)
                evidences.append(np.nan)
                evidence_errs.append(np.nan)
        evidence_dict[recovery_mode] = np.array(evidences)
        evidence_err_dict[recovery_mode] = np.array(evidence_errs)


    for k, v in evidence_dict.items():
        if k == 'pure_qpo':
            label = 'qpo'
        elif k == 'qpo':
            continue
            label = 'red_noise + qpo eq. amp.'
        elif k == 'general_qpo':
            label = 'red_noise + qpo'
        else:
            label = k

        color_dict = dict(general_qpo="blue", red_noise="red")
        linestyle_dict = dict(fred='solid', skew_gaussian='dotted', fred_norris='dashed', fred_norris_extended='dashdot')
        plt.plot(np.arange(len(v)), np.array(v), label=f"{mean_model} {label}", color=color_dict[k],
                 linestyle=linestyle_dict[mean_model])

plt.xlabel('N FRED components')
plt.ylabel(f'ln Z')
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[0, 1, 2, 3, 4])
plt.legend()
plt.savefig(f'GRB_Ln_BF_plot_rwalk_combined.png')
plt.show()
