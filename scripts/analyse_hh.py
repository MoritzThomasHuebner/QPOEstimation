import QPOEstimation
import numpy as np
import bilby

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")

import os
flares = np.array(os.listdir('hares_and_hounds_HH2'))

evidence_qpo_1_fred_list = []
evidence_qpo_2_fred_list = []
evidence_qpo_1_gaussian_list = []
evidence_red_noise_1_fred_list = []
evidence_red_noise_2_fred_list = []
evidence_red_noise_1_gaussian_list = []
run_mode = 'from_maximum'

try:
    evidence_qpo_1_fred_list = np.loadtxt('hh_evidence_qpo_1_fred')
    evidence_qpo_2_fred_list = np.loadtxt('hh_evidence_qpo_2_fred')
    evidence_qpo_1_gaussian_list = np.loadtxt('hh_evidence_qpo_1_gaussian')
    evidence_red_noise_1_fred_list = np.loadtxt('hh_evidence_red_noise_1_fred')
    evidence_red_noise_2_fred_list = np.loadtxt('hh_evidence_red_noise_2_fred')
    evidence_red_noise_1_gaussian_list = np.loadtxt('hh_evidence_red_noise_1_gaussian')
except Exception:
    for flare in flares:
        print(flare)
        try:
            res_qpo_1_fred = QPOEstimation.result.GPResult.from_json(filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/general_qpo/gaussian_process/results/{run_mode}_1_freds_result.json')
            res_qpo_2_fred = QPOEstimation.result.GPResult.from_json(filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/general_qpo/gaussian_process/results/{run_mode}_2_freds_result.json')
            res_qpo_1_gaussian = QPOEstimation.result.GPResult.from_json(filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/general_qpo/gaussian_process/results/{run_mode}_1_gaussians_result.json')
            res_red_noise_1_fred = QPOEstimation.result.GPResult.from_json(filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/gaussian_process/results/{run_mode}_1_freds_result.json')
            res_red_noise_2_fred = QPOEstimation.result.GPResult.from_json(filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/gaussian_process/results/{run_mode}_2_freds_result.json')
            res_red_noise_1_gaussian = QPOEstimation.result.GPResult.from_json(filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/gaussian_process/results/{run_mode}_1_gaussians_result.json')

            evidence_qpo_1_fred = res_qpo_1_fred.res_qpo_1_fred
            evidence_qpo_2_fred = res_qpo_1_fred.res_qpo_2_fred
            evidence_qpo_1_gaussian = res_qpo_1_fred.res_qpo_1_gaussian
            evidence_red_noise_1_fred = res_qpo_1_fred.res_red_noise_1_fred
            evidence_red_noise_2_fred = res_qpo_1_fred.res_red_noise_2_fred
            evidence_red_noise_1_gaussian = res_qpo_1_fred.res_red_noise_1_gaussian
        except Exception:
            evidence_qpo_1_fred = np.nan
            evidence_qpo_2_fred = np.nan
            evidence_qpo_1_gaussian = np.nan
            evidence_red_noise_1_fred = np.nan
            evidence_red_noise_2_fred = np.nan
            evidence_red_noise_1_gaussian = np.nan

        evidence_qpo_1_fred_list.append(evidence_qpo_1_fred)
        evidence_qpo_2_fred_list.append(evidence_qpo_2_fred)
        evidence_qpo_1_gaussian_list.append(evidence_qpo_1_gaussian)
        evidence_red_noise_1_fred_list.append(evidence_red_noise_1_fred)
        evidence_red_noise_2_fred_list.append(evidence_red_noise_2_fred)
        evidence_red_noise_1_gaussian_list.append(evidence_red_noise_1_gaussian)

    evidence_qpo_1_fred_list = np.array(evidence_qpo_1_fred_list)
    evidence_qpo_2_fred_list = np.array(evidence_qpo_2_fred_list)
    evidence_qpo_1_gaussian_list = np.array(evidence_qpo_1_gaussian_list)
    evidence_red_noise_1_fred_list = np.array(evidence_red_noise_1_fred_list)
    evidence_red_noise_2_fred_list = np.array(evidence_red_noise_2_fred_list)
    evidence_red_noise_1_gaussian_list = np.array(evidence_red_noise_1_gaussian_list)

    np.savetxt('hh_evidence_qpo_1_fred', evidence_qpo_1_fred_list)
    np.savetxt('hh_evidence_qpo_2_fred', evidence_qpo_2_fred_list)
    np.savetxt('hh_evidence_qpo_1_gaussian', evidence_qpo_1_gaussian_list)
    np.savetxt('hh_evidence_red_noise_1_fred', evidence_red_noise_1_fred_list)
    np.savetxt('hh_evidence_red_noise_2_fred', evidence_red_noise_2_fred_list)
    np.savetxt('hh_evidence_red_noise_1_gaussian', evidence_red_noise_1_gaussian_list)


for red_noise_evidences, qpo_evidences in zip([evidence_red_noise_1_fred_list, evidence_red_noise_2_fred_list, evidence_red_noise_1_gaussian_list],
                                              [evidence_qpo_1_fred_list, evidence_qpo_2_fred_list, evidence_qpo_1_gaussian_list]):

    qpo_candidates = np.where(qpo_evidences - red_noise_evidences > 1)[0]
    plt.plot(qpo_evidences - red_noise_evidences)
    plt.show()
    print(qpo_candidates)
    print(qpo_evidences[qpo_candidates] - red_noise_evidences[qpo_candidates])
    print(flares[qpo_candidates])
    print()
