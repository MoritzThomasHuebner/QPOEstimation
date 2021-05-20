import QPOEstimation
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("Qt5Agg")
import pandas as pd

import os

flares = np.array(sorted(os.listdir('hares_and_hounds_HH2_just_figures')))
print(len(flares))
run_mode = 'from_maximum'

results = np.genfromtxt("data/hares_and_hounds/qpp_type_hh2.csv", delimiter=',', dtype=int)
flare_keys = results[:, 0]
flare_types = results[:, 1]

threshold_ln_bf = 0.24

try:
    evidence_qpo_1_fred_list = np.loadtxt('hh_evidence_qpo_1_fred')
    evidence_qpo_2_fred_list = np.loadtxt('hh_evidence_qpo_2_fred')
    evidence_qpo_3_fred_list = np.loadtxt('hh_evidence_qpo_3_fred')
    evidence_qpo_1_gaussian_list = np.loadtxt('hh_evidence_qpo_1_gaussian')
    evidence_qpo_2_gaussian_list = np.loadtxt('hh_evidence_qpo_2_gaussian')
    evidence_qpo_3_gaussian_list = np.loadtxt('hh_evidence_qpo_3_gaussian')
    evidence_red_noise_1_fred_list = np.loadtxt('hh_evidence_red_noise_1_fred')
    evidence_red_noise_2_fred_list = np.loadtxt('hh_evidence_red_noise_2_fred')
    evidence_red_noise_3_fred_list = np.loadtxt('hh_evidence_red_noise_3_fred')
    evidence_red_noise_1_gaussian_list = np.loadtxt('hh_evidence_red_noise_1_gaussian')
    evidence_red_noise_2_gaussian_list = np.loadtxt('hh_evidence_red_noise_2_gaussian')
    evidence_red_noise_3_gaussian_list = np.loadtxt('hh_evidence_red_noise_3_gaussian')
except Exception:
    evidence_qpo_1_fred_list = []
    evidence_qpo_2_fred_list = []
    evidence_qpo_3_fred_list = []
    evidence_qpo_1_gaussian_list = []
    evidence_qpo_2_gaussian_list = []
    evidence_qpo_3_gaussian_list = []
    evidence_red_noise_1_fred_list = []
    evidence_red_noise_2_fred_list = []
    evidence_red_noise_3_fred_list = []
    evidence_red_noise_1_gaussian_list = []
    evidence_red_noise_2_gaussian_list = []
    evidence_red_noise_3_gaussian_list = []
    for flare in flares:
        print(flare)
        try:
            res_qpo_1_fred = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/general_qpo/gaussian_process/results/{run_mode}_1_freds_result.json')
            res_qpo_2_fred = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/general_qpo/gaussian_process/results/{run_mode}_2_freds_result.json')
            res_qpo_3_fred = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/general_qpo/gaussian_process/results/{run_mode}_3_freds_result.json')
            res_qpo_1_gaussian = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/general_qpo/gaussian_process/results/{run_mode}_1_gaussians_result.json')
            res_qpo_2_gaussian = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/general_qpo/gaussian_process/results/{run_mode}_2_gaussians_result.json')
            res_qpo_3_gaussian = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/general_qpo/gaussian_process/results/{run_mode}_3_gaussians_result.json')
            res_red_noise_1_fred = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/gaussian_process/results/{run_mode}_1_freds_result.json')
            res_red_noise_2_fred = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/gaussian_process/results/{run_mode}_2_freds_result.json')
            res_red_noise_3_fred = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/gaussian_process/results/{run_mode}_3_freds_result.json')
            res_red_noise_1_gaussian = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/gaussian_process/results/{run_mode}_1_gaussians_result.json')
            res_red_noise_2_gaussian = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/gaussian_process/results/{run_mode}_2_gaussians_result.json')
            res_red_noise_3_gaussian = QPOEstimation.result.GPResult.from_json(
                filename=f'hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/gaussian_process/results/{run_mode}_3_gaussians_result.json')

            evidence_qpo_1_fred = res_qpo_1_fred.log_evidence
            evidence_qpo_2_fred = res_qpo_2_fred.log_evidence
            evidence_qpo_3_fred = res_qpo_3_fred.log_evidence
            evidence_qpo_1_gaussian = res_qpo_1_gaussian.log_evidence
            evidence_qpo_2_gaussian = res_qpo_2_gaussian.log_evidence
            evidence_qpo_3_gaussian = res_qpo_3_gaussian.log_evidence
            evidence_red_noise_1_fred = res_red_noise_1_fred.log_evidence
            evidence_red_noise_2_fred = res_red_noise_2_fred.log_evidence
            evidence_red_noise_3_fred = res_red_noise_3_fred.log_evidence
            evidence_red_noise_1_gaussian = res_red_noise_1_gaussian.log_evidence
            evidence_red_noise_2_gaussian = res_red_noise_2_gaussian.log_evidence
            evidence_red_noise_3_gaussian = res_red_noise_3_gaussian.log_evidence
        except Exception:
            evidence_qpo_1_fred = np.nan
            evidence_qpo_2_fred = np.nan
            evidence_qpo_3_fred = np.nan
            evidence_qpo_1_gaussian = np.nan
            evidence_qpo_2_gaussian = np.nan
            evidence_qpo_3_gaussian = np.nan
            evidence_red_noise_1_fred = np.nan
            evidence_red_noise_2_fred = np.nan
            evidence_red_noise_3_fred = np.nan
            evidence_red_noise_1_gaussian = np.nan
            evidence_red_noise_2_gaussian = np.nan
            evidence_red_noise_3_gaussian = np.nan

        evidence_qpo_1_fred_list.append(evidence_qpo_1_fred)
        evidence_qpo_2_fred_list.append(evidence_qpo_2_fred)
        evidence_qpo_3_fred_list.append(evidence_qpo_3_fred)
        evidence_qpo_1_gaussian_list.append(evidence_qpo_1_gaussian)
        evidence_qpo_2_gaussian_list.append(evidence_qpo_2_gaussian)
        evidence_qpo_3_gaussian_list.append(evidence_qpo_3_gaussian)
        evidence_red_noise_1_fred_list.append(evidence_red_noise_1_fred)
        evidence_red_noise_2_fred_list.append(evidence_red_noise_2_fred)
        evidence_red_noise_3_fred_list.append(evidence_red_noise_3_fred)
        evidence_red_noise_1_gaussian_list.append(evidence_red_noise_1_gaussian)
        evidence_red_noise_2_gaussian_list.append(evidence_red_noise_2_gaussian)
        evidence_red_noise_3_gaussian_list.append(evidence_red_noise_3_gaussian)

    evidence_qpo_1_fred_list = np.array(evidence_qpo_1_fred_list)
    evidence_qpo_2_fred_list = np.array(evidence_qpo_2_fred_list)
    evidence_qpo_3_fred_list = np.array(evidence_qpo_3_fred_list)
    evidence_qpo_1_gaussian_list = np.array(evidence_qpo_1_gaussian_list)
    evidence_qpo_2_gaussian_list = np.array(evidence_qpo_2_gaussian_list)
    evidence_qpo_3_gaussian_list = np.array(evidence_qpo_3_gaussian_list)
    evidence_red_noise_1_fred_list = np.array(evidence_red_noise_1_fred_list)
    evidence_red_noise_2_fred_list = np.array(evidence_red_noise_2_fred_list)
    evidence_red_noise_3_fred_list = np.array(evidence_red_noise_3_fred_list)
    evidence_red_noise_1_gaussian_list = np.array(evidence_red_noise_1_gaussian_list)
    evidence_red_noise_2_gaussian_list = np.array(evidence_red_noise_2_gaussian_list)
    evidence_red_noise_3_gaussian_list = np.array(evidence_red_noise_3_gaussian_list)

    np.savetxt('hh_evidence_qpo_1_fred', evidence_qpo_1_fred_list)
    np.savetxt('hh_evidence_qpo_2_fred', evidence_qpo_2_fred_list)
    np.savetxt('hh_evidence_qpo_3_fred', evidence_qpo_3_fred_list)
    np.savetxt('hh_evidence_qpo_1_gaussian', evidence_qpo_1_gaussian_list)
    np.savetxt('hh_evidence_qpo_2_gaussian', evidence_qpo_2_gaussian_list)
    np.savetxt('hh_evidence_qpo_3_gaussian', evidence_qpo_3_gaussian_list)
    np.savetxt('hh_evidence_red_noise_1_fred', evidence_red_noise_1_fred_list)
    np.savetxt('hh_evidence_red_noise_2_fred', evidence_red_noise_2_fred_list)
    np.savetxt('hh_evidence_red_noise_3_fred', evidence_red_noise_3_fred_list)
    np.savetxt('hh_evidence_red_noise_1_gaussian', evidence_red_noise_1_gaussian_list)
    np.savetxt('hh_evidence_red_noise_2_gaussian', evidence_red_noise_2_gaussian_list)
    np.savetxt('hh_evidence_red_noise_3_gaussian', evidence_red_noise_3_gaussian_list)

for red_noise_evidences, qpo_evidences, label in zip(
        [evidence_qpo_1_fred_list, evidence_qpo_2_fred_list, evidence_qpo_3_fred_list, evidence_qpo_1_gaussian_list,
         evidence_qpo_2_gaussian_list, evidence_qpo_3_gaussian_list],
        [evidence_red_noise_1_fred_list, evidence_red_noise_2_fred_list, evidence_red_noise_3_fred_list,
         evidence_red_noise_1_gaussian_list, evidence_red_noise_2_gaussian_list, evidence_red_noise_3_gaussian_list],
        ['1_freds', '2_freds', '3_freds', '1_gaussians', '2_gaussians', '3_gaussians']):
    ln_bfs_i = qpo_evidences - red_noise_evidences
    ln_bfs_1_fred = evidence_red_noise_1_fred_list - evidence_qpo_1_fred_list
    plt.plot([x for _, x in sorted(zip(ln_bfs_1_fred, ln_bfs_i))][::-1], label=label)
    qpo_candidates = np.where(qpo_evidences - red_noise_evidences > threshold_ln_bf)[0]
    print(qpo_candidates)
    print(qpo_evidences[qpo_candidates] - red_noise_evidences[qpo_candidates])
    print(flares[qpo_candidates])
    print()

plt.xlabel('(Sorted) Event ID')
plt.ylabel('QPO ln BF')
plt.legend()
plt.savefig('hh_qpo_bayes_factors.png')
plt.clf()

ln_bfs_1_fred = evidence_qpo_1_fred_list - evidence_qpo_1_fred_list
ln_bfs_2_fred = evidence_qpo_2_fred_list - evidence_qpo_1_fred_list
ln_bfs_3_fred = evidence_qpo_3_fred_list - evidence_qpo_1_fred_list
ln_bfs_1_gaussian = evidence_qpo_1_gaussian_list - evidence_qpo_1_fred_list
ln_bfs_2_gaussian = evidence_qpo_2_gaussian_list - evidence_qpo_1_fred_list
ln_bfs_3_gaussian = evidence_qpo_3_gaussian_list - evidence_qpo_1_fred_list
for ln_bfs, label in zip(
        [ln_bfs_1_fred, ln_bfs_2_fred, ln_bfs_3_fred, ln_bfs_1_gaussian, ln_bfs_2_gaussian, ln_bfs_3_gaussian],
        ['1_freds', '2_freds', '3_freds', '1_gaussians', '2_gaussians', '3_gaussians']):
    plt.plot([x for _, x in sorted(zip(ln_bfs_1_fred, ln_bfs))], label=label)
plt.xlabel('(Sorted) Event ID')
plt.ylabel('Mean model ln BF')
plt.legend()
plt.savefig('hh_mean_model_qpo_bayes_factors.png')
plt.clf()

ln_bfs_1_fred = evidence_red_noise_1_fred_list - evidence_red_noise_1_fred_list
ln_bfs_2_fred = evidence_red_noise_2_fred_list - evidence_red_noise_1_fred_list
ln_bfs_3_fred = evidence_red_noise_3_fred_list - evidence_red_noise_1_fred_list
ln_bfs_1_gaussian = evidence_red_noise_1_gaussian_list - evidence_red_noise_1_fred_list
ln_bfs_2_gaussian = evidence_red_noise_2_gaussian_list - evidence_red_noise_1_fred_list
ln_bfs_3_gaussian = evidence_red_noise_3_gaussian_list - evidence_red_noise_1_fred_list

for ln_bfs, label in zip(
        [ln_bfs_1_fred, ln_bfs_2_fred, ln_bfs_3_fred, ln_bfs_1_gaussian, ln_bfs_2_gaussian, ln_bfs_3_gaussian],
        ['1_freds', '2_freds', '3_freds', '1_gaussians', '2_gaussians', '3_gaussians']):
    plt.plot([x for _, x in sorted(zip(ln_bfs_1_fred, ln_bfs))], label=label)

plt.xlabel('Event ID')
plt.ylabel('Mean model ln BF')
plt.legend()
plt.savefig('hh_mean_model_red_noise_bayes_factors.png')
plt.clf()

qpo_evidence_list_list = [evidence_qpo_1_fred_list, evidence_qpo_2_fred_list, evidence_qpo_3_fred_list,
                          evidence_qpo_1_gaussian_list, evidence_qpo_2_gaussian_list, evidence_qpo_3_gaussian_list]
red_noise_evidence_list_list = [evidence_red_noise_1_fred_list, evidence_red_noise_2_fred_list,
                                evidence_red_noise_3_fred_list, evidence_red_noise_1_gaussian_list,
                                evidence_red_noise_2_gaussian_list, evidence_red_noise_3_gaussian_list]
qpo_max_evidence_tags = []
red_noise_max_evidence_tags = []
for i in range(len(evidence_qpo_1_fred_list)):
    qpo_max_evidence_tags.append(np.argmax(
        [evidence_qpo_1_fred_list[i], evidence_qpo_2_fred_list[i], evidence_qpo_3_fred_list[i],
         evidence_qpo_1_gaussian_list[i], evidence_qpo_2_gaussian_list[i], evidence_qpo_3_gaussian_list[i]]))
    red_noise_max_evidence_tags.append(np.argmax(
        [evidence_red_noise_1_fred_list[i], evidence_red_noise_2_fred_list[i], evidence_red_noise_3_fred_list[i],
         evidence_red_noise_1_gaussian_list[i], evidence_red_noise_2_gaussian_list[i],
         evidence_red_noise_3_gaussian_list[i]]))

qpo_max_evidence_tags = np.array(qpo_max_evidence_tags)
red_noise_max_evidence_tags = np.array(red_noise_max_evidence_tags)

qpo_evidences = []
red_noise_evidences = []

for i in range(len(qpo_max_evidence_tags)):
    qpo_evidences.append(qpo_evidence_list_list[qpo_max_evidence_tags[i]][i])
    red_noise_evidences.append(red_noise_evidence_list_list[red_noise_max_evidence_tags[i]][i])

qpo_evidences = np.array(qpo_evidences)
red_noise_evidences = np.array(red_noise_evidences)

plt.plot(qpo_evidences - red_noise_evidences)
plt.xlabel('Event ID')
plt.ylabel('QPO ln BF')
plt.legend()
plt.savefig('hh_qpo_bayes_factors_optimal_mean.png')
plt.clf()

qpo_candidates = np.where(qpo_evidences - red_noise_evidences > threshold_ln_bf)[0]

tag_table = {0: "1 FREDs", 1: "2 FREDs", 2: "3 FREDs", 3: "1 Gauss", 4: "2 Gauss", 5: "3 Gauss"}

injection_types = {
    10: "simple qpp, 1 exponential",
    20: "simple qpp, 1 gaussian",
    40: "no qpp, 1 exponential",
    50: "no qpp, 1 gaussian",
    41: "no qpp, 2 exponential",
    51: "no qpp, 2 gaussian",
    42: "3 periodic exponential flares",
    52: "3 periodic gaussian flares",
    43: "no qpp, 3 exponential",
    53: "no qpp, 3 gaussian"}

print("Flare ID\tln BF\tRN mean model\tQPO mean model\tTrue Model")
for qpo_candidate, qpo_evidence, red_noise_evidence, flare, qpo_max_evidence_tag, red_noise_max_evidence_tag in zip(
        qpo_candidates, qpo_evidences[qpo_candidates], red_noise_evidences[qpo_candidates], flares[qpo_candidates],
        qpo_max_evidence_tags[qpo_candidates], red_noise_max_evidence_tags[qpo_candidates]):
    flare_type = flare_types[np.where(flare_keys == int(flare))[0]]
    ln_bf = "{:.3f}".format(qpo_evidence - red_noise_evidence)
    print(f"{flare}\t\t{ln_bf}\t{tag_table[qpo_max_evidence_tag]}\t\t\t"
          f"{tag_table[red_noise_max_evidence_tag]}\t\t\t{injection_types[flare_type[0]]}")
