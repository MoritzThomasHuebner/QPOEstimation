import QPOEstimation
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("Qt5Agg")
import pandas as pd

import os

flares = np.array(sorted(os.listdir("results/hares_and_hounds_HH2")))
print(len(flares))
run_mode = "from_maximum"

# results = np.genfromtxt("data/hares_and_hounds/qpp_type_hh2.csv", delimiter=",", dtype=int)
results = np.loadtxt("data/hares_and_hounds/qpp_type_hh2.txt")
flare_keys = results[:, 0]
# flare_keys = [int(k) for k in flare_keys]
flare_types = results[:, 1]

threshold_ln_bf = -1.0

try:
    evidence_qpo_1_skew_exponential_list = np.loadtxt("results/hh_evidence_qpo_1_skew_exponential")
    # evidence_qpo_2_skew_exponential_list = np.loadtxt("results/hh_evidence_qpo_2_skew_exponential")
    # evidence_qpo_3_skew_exponential_list = np.loadtxt("hh_evidence_qpo_3_skew_exponential")
    # evidence_qpo_1_gaussian_list = np.loadtxt("hh_evidence_qpo_1_gaussian")
    # evidence_qpo_2_gaussian_list = np.loadtxt("hh_evidence_qpo_2_gaussian")
    # evidence_qpo_3_gaussian_list = np.loadtxt("hh_evidence_qpo_3_gaussian")
    evidence_red_noise_1_skew_exponential_list = np.loadtxt("results/hh_evidence_red_noise_1_skew_exponential")
    # evidence_red_noise_2_skew_exponential_list = np.loadtxt("results/hh_evidence_red_noise_2_skew_exponential")
    # evidence_red_noise_3_skew_exponential_list = np.loadtxt("hh_evidence_red_noise_3_skew_exponential")
    # evidence_red_noise_1_gaussian_list = np.loadtxt("hh_evidence_red_noise_1_gaussian")
    # evidence_red_noise_2_gaussian_list = np.loadtxt("hh_evidence_red_noise_2_gaussian")
    # evidence_red_noise_3_gaussian_list = np.loadtxt("hh_evidence_red_noise_3_gaussian")
except Exception:
    evidence_qpo_1_skew_exponential_list = []
    # evidence_qpo_2_skew_exponential_list = []
    # evidence_qpo_3_skew_exponential_list = []
    # evidence_qpo_1_gaussian_list = []
    # evidence_qpo_2_gaussian_list = []
    # evidence_qpo_3_gaussian_list = []
    evidence_red_noise_1_skew_exponential_list = []
    # evidence_red_noise_2_skew_exponential_list = []
    # evidence_red_noise_3_skew_exponential_list = []
    # evidence_red_noise_1_gaussian_list = []
    # evidence_red_noise_2_gaussian_list = []
    # evidence_red_noise_3_gaussian_list = []
    for flare in flares:
        print(flare)
        try:
            res_qpo_1_skew_exponential = QPOEstimation.result.GPResult.from_json(
                filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/qpo_plus_red_noise/celerite/results/{run_mode}_1_skew_exponentials_result.json")
            # res_qpo_2_skew_exponential = QPOEstimation.result.GPResult.from_json(
            #     filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/qpo_plus_red_noise/celerite/results/{run_mode}_2_skew_exponentials_result.json")
            # res_qpo_3_skew_exponential = QPOEstimation.result.GPResult.from_json(
            #     filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/qpo_plus_red_noise/celerite/results/{run_mode}_3_skew_exponentials_result.json")
            # res_qpo_1_gaussian = QPOEstimation.result.GPResult.from_json(
            #     filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/qpo_plus_red_noise/celerite/results/{run_mode}_1_gaussians_result.json")
            # res_qpo_2_gaussian = QPOEstimation.result.GPResult.from_json(
            #     filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/qpo_plus_red_noise/celerite/results/{run_mode}_2_gaussians_result.json")
            # res_qpo_3_gaussian = QPOEstimation.result.GPResult.from_json(
            #     filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/qpo_plus_red_noise/celerite/results/{run_mode}_3_gaussians_result.json")
            res_red_noise_1_skew_exponential = QPOEstimation.result.GPResult.from_json(
                filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/celerite/results/{run_mode}_1_skew_exponentials_result.json")
            # res_red_noise_2_skew_exponential = QPOEstimation.result.GPResult.from_json(
            #     filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/celerite/results/{run_mode}_2_skew_exponentials_result.json")
            # res_red_noise_3_skew_exponential = QPOEstimation.result.GPResult.from_json(
            #     filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/celerite/results/{run_mode}_3_skew_exponentials_result.json")
            # res_red_noise_1_gaussian = QPOEstimation.result.GPResult.from_json(
            #     filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/celerite/results/{run_mode}_1_gaussians_result.json")
            # res_red_noise_2_gaussian = QPOEstimation.result.GPResult.from_json(
            #     filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/celerite/results/{run_mode}_2_gaussians_result.json")
            # res_red_noise_3_gaussian = QPOEstimation.result.GPResult.from_json(
            #     filename=f"results/hares_and_hounds_HH2/{flare}/{run_mode}/red_noise/celerite/results/{run_mode}_3_gaussians_result.json")

            evidence_qpo_1_skew_exponential = res_qpo_1_skew_exponential.log_evidence
            # evidence_qpo_2_skew_exponential = res_qpo_2_skew_exponential.log_evidence
            # evidence_qpo_3_skew_exponential = res_qpo_3_skew_exponential.log_evidence
            # evidence_qpo_1_gaussian = res_qpo_1_gaussian.log_evidence
            # evidence_qpo_2_gaussian = res_qpo_2_gaussian.log_evidence
            # evidence_qpo_3_gaussian = res_qpo_3_gaussian.log_evidence
            evidence_red_noise_1_skew_exponential = res_red_noise_1_skew_exponential.log_evidence
            # evidence_red_noise_2_skew_exponential = res_red_noise_2_skew_exponential.log_evidence
            # evidence_red_noise_3_skew_exponential = res_red_noise_3_skew_exponential.log_evidence
            # evidence_red_noise_1_gaussian = res_red_noise_1_gaussian.log_evidence
            # evidence_red_noise_2_gaussian = res_red_noise_2_gaussian.log_evidence
            # evidence_red_noise_3_gaussian = res_red_noise_3_gaussian.log_evidence
        except Exception as e:
            evidence_qpo_1_skew_exponential = np.nan
            # evidence_qpo_2_skew_exponential = np.nan
            # evidence_qpo_3_skew_exponential = np.nan
            # evidence_qpo_1_gaussian = np.nan
            # evidence_qpo_2_gaussian = np.nan
            # evidence_qpo_3_gaussian = np.nan
            evidence_red_noise_1_skew_exponential = np.nan
            # evidence_red_noise_2_skew_exponential = np.nan
            # evidence_red_noise_3_skew_exponential = np.nan
            # evidence_red_noise_1_gaussian = np.nan
            # evidence_red_noise_2_gaussian = np.nan
            # evidence_red_noise_3_gaussian = np.nan

        evidence_qpo_1_skew_exponential_list.append(evidence_qpo_1_skew_exponential)
        # evidence_qpo_2_skew_exponential_list.append(evidence_qpo_2_skew_exponential)
        # evidence_qpo_3_skew_exponential_list.append(evidence_qpo_3_skew_exponential)
        # evidence_qpo_1_gaussian_list.append(evidence_qpo_1_gaussian)
        # evidence_qpo_2_gaussian_list.append(evidence_qpo_2_gaussian)
        # evidence_qpo_3_gaussian_list.append(evidence_qpo_3_gaussian)
        evidence_red_noise_1_skew_exponential_list.append(evidence_red_noise_1_skew_exponential)
        # evidence_red_noise_2_skew_exponential_list.append(evidence_red_noise_2_skew_exponential)
        # evidence_red_noise_3_skew_exponential_list.append(evidence_red_noise_3_skew_exponential)
        # evidence_red_noise_1_gaussian_list.append(evidence_red_noise_1_gaussian)
        # evidence_red_noise_2_gaussian_list.append(evidence_red_noise_2_gaussian)
        # evidence_red_noise_3_gaussian_list.append(evidence_red_noise_3_gaussian)

    evidence_qpo_1_skew_exponential_list = np.array(evidence_qpo_1_skew_exponential_list)
    # evidence_qpo_2_skew_exponential_list = np.array(evidence_qpo_2_skew_exponential_list)
    # evidence_qpo_3_skew_exponential_list = np.array(evidence_qpo_3_skew_exponential_list)
    # evidence_qpo_1_gaussian_list = np.array(evidence_qpo_1_gaussian_list)
    # evidence_qpo_2_gaussian_list = np.array(evidence_qpo_2_gaussian_list)
    # evidence_qpo_3_gaussian_list = np.array(evidence_qpo_3_gaussian_list)
    evidence_red_noise_1_skew_exponential_list = np.array(evidence_red_noise_1_skew_exponential_list)
    # evidence_red_noise_2_skew_exponential_list = np.array(evidence_red_noise_2_skew_exponential_list)
    # evidence_red_noise_3_skew_exponential_list = np.array(evidence_red_noise_3_skew_exponential_list)
    # evidence_red_noise_1_gaussian_list = np.array(evidence_red_noise_1_gaussian_list)
    # evidence_red_noise_2_gaussian_list = np.array(evidence_red_noise_2_gaussian_list)
    # evidence_red_noise_3_gaussian_list = np.array(evidence_red_noise_3_gaussian_list)

    np.savetxt("results/hh_evidence_qpo_1_skew_exponential", evidence_qpo_1_skew_exponential_list)
    # np.savetxt("results/hh_evidence_qpo_2_skew_exponential", evidence_qpo_2_skew_exponential_list)
    # np.savetxt("results/hh_evidence_qpo_3_skew_exponential", evidence_qpo_3_skew_exponential_list)
    # np.savetxt("results/hh_evidence_qpo_1_gaussian", evidence_qpo_1_gaussian_list)
    # np.savetxt("results/hh_evidence_qpo_2_gaussian", evidence_qpo_2_gaussian_list)
    # np.savetxt("results/hh_evidence_qpo_3_gaussian", evidence_qpo_3_gaussian_list)
    np.savetxt("results/hh_evidence_red_noise_1_skew_exponential", evidence_red_noise_1_skew_exponential_list)
    # np.savetxt("results/hh_evidence_red_noise_2_skew_exponential", evidence_red_noise_2_skew_exponential_list)
    # np.savetxt("results/hh_evidence_red_noise_3_skew_exponential", evidence_red_noise_3_skew_exponential_list)
    # np.savetxt("results/hh_evidence_red_noise_1_gaussian", evidence_red_noise_1_gaussian_list)
    # np.savetxt("results/hh_evidence_red_noise_2_gaussian", evidence_red_noise_2_gaussian_list)
    # np.savetxt("results/hh_evidence_red_noise_3_gaussian", evidence_red_noise_3_gaussian_list)

for red_noise_evidences, qpo_evidences, label in zip(
        [evidence_qpo_1_skew_exponential_list],
        [evidence_red_noise_1_skew_exponential_list],
        ["1_skew_exponentials", "2_skew_exponentials"]):
    ln_bfs_i = qpo_evidences - red_noise_evidences
    ln_bfs_1_skew_exponential = evidence_red_noise_1_skew_exponential_list - evidence_qpo_1_skew_exponential_list
    plt.plot([x for _, x in sorted(zip(ln_bfs_1_skew_exponential, ln_bfs_i))][::-1], label=label)
    qpo_candidates = np.where(qpo_evidences - red_noise_evidences > threshold_ln_bf)[0]
    print(qpo_candidates)
    print(qpo_evidences[qpo_candidates] - red_noise_evidences[qpo_candidates])
    print(flares[qpo_candidates])
    print()

plt.xlabel("(Sorted) Event ID")
plt.ylabel("QPO ln BF")
plt.legend()
plt.savefig("results/hh_qpo_bayes_factors.png")
plt.clf()

ln_bfs_1_skew_exponential = evidence_qpo_1_skew_exponential_list - evidence_qpo_1_skew_exponential_list
# ln_bfs_2_skew_exponential = evidence_qpo_2_skew_exponential_list - evidence_qpo_1_skew_exponential_list
# ln_bfs_3_skew_exponential = evidence_qpo_3_skew_exponential_list - evidence_qpo_1_skew_exponential_list
# ln_bfs_1_gaussian = evidence_qpo_1_gaussian_list - evidence_qpo_1_skew_exponential_list
# ln_bfs_2_gaussian = evidence_qpo_2_gaussian_list - evidence_qpo_1_skew_exponential_list
# ln_bfs_3_gaussian = evidence_qpo_3_gaussian_list - evidence_qpo_1_skew_exponential_list
for ln_bfs, label in zip(
        [ln_bfs_1_skew_exponential],
        ["1_skew_exponentials", "2_skew_exponentials"]):
    plt.plot([x for _, x in sorted(zip(ln_bfs_1_skew_exponential, ln_bfs))], label=label)
plt.xlabel("(Sorted) Event ID")
plt.ylabel("Mean model ln BF")
plt.legend()
plt.savefig("results/hh_mean_model_qpo_bayes_factors.png")
plt.clf()

ln_bfs_1_skew_exponential = evidence_red_noise_1_skew_exponential_list - evidence_red_noise_1_skew_exponential_list
# ln_bfs_2_skew_exponential = evidence_red_noise_2_skew_exponential_list - evidence_red_noise_1_skew_exponential_list
# ln_bfs_3_skew_exponential = evidence_red_noise_3_skew_exponential_list - evidence_red_noise_1_skew_exponential_list
# ln_bfs_1_gaussian = evidence_red_noise_1_gaussian_list - evidence_red_noise_1_skew_exponential_list
# ln_bfs_2_gaussian = evidence_red_noise_2_gaussian_list - evidence_red_noise_1_skew_exponential_list
# ln_bfs_3_gaussian = evidence_red_noise_3_gaussian_list - evidence_red_noise_1_skew_exponential_list

for ln_bfs, label in zip(
        [ln_bfs_1_skew_exponential],
        ["1_skew_exponentials", "2_skew_exponentials", "3_skew_exponentials"]):
    plt.plot([x for _, x in sorted(zip(ln_bfs_1_skew_exponential, ln_bfs))], label=label)

plt.xlabel("Event ID")
plt.ylabel("Mean model ln BF")
plt.legend()
plt.savefig("results/hh_mean_model_red_noise_bayes_factors.png")
plt.clf()

qpo_evidence_list_list = [evidence_qpo_1_skew_exponential_list]
red_noise_evidence_list_list = [evidence_red_noise_1_skew_exponential_list]
qpo_max_evidence_tags = []
red_noise_max_evidence_tags = []
for i in range(len(evidence_qpo_1_skew_exponential_list)):
    qpo_max_evidence_tags.append(np.argmax(
        [evidence_qpo_1_skew_exponential_list[i]]))
    red_noise_max_evidence_tags.append(np.argmax(
        [evidence_red_noise_1_skew_exponential_list[i]]))

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
plt.xlabel("Event ID")
plt.ylabel("QPO ln BF")
plt.legend()
plt.savefig("results/hh_qpo_bayes_factors_optimal_mean.png")
plt.clf()

qpo_candidates = np.where(qpo_evidences - red_noise_evidences > threshold_ln_bf)[0]

tag_table = {0: "1 Skew Exps.", 1: "2 Skew Exps.", 2: "3 Skew Exps.", 3: "1 Gauss", 4: "2 Gauss", 5: "3 Gauss"}

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
