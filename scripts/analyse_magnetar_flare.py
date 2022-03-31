import numpy as np
import bilby
from QPOEstimation.result import GPResult, power_qpo
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Qt5Agg")
plt.style.use("paper.mplstyle")

evidence_dict = dict()
evidence_err_dict = dict()

res_dict = {}

linestyle_dict = dict(skew_exponential="solid", skew_gaussian="dotted", fred="dashed")
label_dict_mean = dict(skew_exponential="skew exp.", fred="FRED", skew_gaussian="skew gaus.")
label_dict_kernel = dict(qpo_plus_red_noise="qpo+rn", red_noise="rn", double_qpo="qpo+qpo")

# plt.figure(figsize=(9.2, 7.2))
# # plt.figure(dpi=150)
for mean_model in ["skew_exponential", "skew_gaussian", "fred"]:
    print(mean_model)
    for recovery_mode in ["qpo_plus_red_noise", "red_noise"]:
        evidences = []
        evidence_errs = []
        res_list = []
        for n_component in range(1, 4):
            try:
                res = GPResult.from_json(
                    outdir=f"results/magnetar_flares/SGR_0501/080823478_lcobs/entire_segment/{recovery_mode}/"
                           f"celerite/results/",
                    label=f"entire_segment_{n_component}_{mean_model}s")
                if recovery_mode == "qpo_plus_red_noise":
                    res_list.append(res)
                evidences.append(res.log_evidence)
                evidence_errs.append(res.log_evidence_err)
                print(f"{recovery_mode}\t{n_component}\t{res.log_evidence}\t{res.posterior.iloc[-1]['log_likelihood']}")
            except Exception as e:
                print(e)
                evidences.append(np.nan)
                evidence_errs.append(np.nan)
        if recovery_mode == "qpo_plus_red_noise":
            res_dict[mean_model] = res_list
        evidence_dict[recovery_mode] = np.array(evidences)
        evidence_err_dict[recovery_mode] = np.array(evidence_errs)
    print()

    for k, v in evidence_dict.items():
        color_dict = dict(qpo_plus_red_noise="blue", red_noise="red")
        plt.plot(np.arange(len(v)), np.array(v), label=f"{label_dict_mean[mean_model]},  {label_dict_kernel[k]}",
                 color=color_dict[k], linestyle=linestyle_dict[mean_model])

plt.xlabel("Number of flare components")
plt.ylabel(f"ln Z")
plt.xticks(ticks=[0, 1, 2], labels=[1, 2, 3])
# plt.xticks(ticks=[0, 1], labels=[1, 2])
# plt.ylim(-1002.25, -980)

# plt.ylim(-372, -357)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"results/Magnetar_Ln_Z_plot.pdf")
# plt.show()
plt.close('all')

# plt.figure(figsize=(9.2, 7.2))
for mean_model in ["skew_exponential", "skew_gaussian", "fred"]:
    print(mean_model)
    for recovery_mode in ["qpo_plus_red_noise", "red_noise"]:
        evidences = []
        evidence_errs = []
        res_list = []
        for n_component in range(1, 4):
            try:
                res = GPResult.from_json(
                    outdir=f"results/magnetar_flares/SGR_0501/080823478_lcobs/entire_segment/{recovery_mode}/"
                           f"celerite/results/",
                    label=f"entire_segment_{n_component}_{mean_model}s")
                if recovery_mode == "qpo_plus_red_noise":
                    res_list.append(res)
                evidences.append(res.log_evidence)
                evidence_errs.append(res.log_evidence_err)
                print(f"{recovery_mode}\t{n_component}\t{res.log_evidence}\t{res.log_evidence_err}")
            except Exception as e:
                print(e)
                evidences.append(np.nan)
                evidence_errs.append(np.nan)
        if recovery_mode == "qpo_plus_red_noise":
            res_dict[mean_model] = res_list
        evidence_dict[recovery_mode] = np.array(evidences)
        evidence_err_dict[recovery_mode] = np.array(evidence_errs)
    print()
    bfs = evidence_dict["qpo_plus_red_noise"] - evidence_dict["red_noise"]
    # bfs_error = np.sqrt(evidence_err_dict["qpo_plus_red_noise"]**2 + evidence_err_dict["red_noise"]**2)
    plt.plot(np.arange(3), bfs, label=f"{label_dict_mean[mean_model]}", linestyle=linestyle_dict[mean_model])

plt.xlabel("Number of flare components")
plt.ylabel(r"$\ln BF_{\mathrm{QPO}}$")
plt.xticks(ticks=[0, 1, 2], labels=[1, 2, 3])
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"results/Magnetar_Ln_BF_plot.pdf")
plt.show()


evidence_dict = dict()
evidence_err_dict = dict()
res_dict = {}
recovery_mode = "qpo_plus_red_noise"
ref_evidence = None

plt.figure(figsize=(9.2, 7.2))
for mean_model in ["skew_exponential", "skew_gaussian", "fred"]:
    evidences = []
    evidence_errs = []
    res_list = []
    for n_component in range(0, 4):
        try:
            res = GPResult.from_json(
                outdir=f"results/magnetar_flares/SGR_0501/080823478_lcobs/entire_segment/qpo_plus_red_noise/"
                       f"celerite/results/",
                label=f"entire_segment_{n_component}_{mean_model}s")
            if ref_evidence is None:
                ref_evidence = res.log_evidence
                print(ref_evidence)
                print(ref_evidence)
                print(ref_evidence)
                print(ref_evidence)
                raise Exception
            res_list.append(res)
            evidences.append(res.log_evidence)
            evidence_errs.append(res.log_evidence_err)
            print(f"{recovery_mode}\t{n_component}\t{res.log_evidence}\t{res.log_evidence_err}")
        except Exception as e:
            print(e)
            evidences.append(np.nan)
            evidence_errs.append(np.nan)
    res_dict[mean_model] = res_list
    evidence_dict[recovery_mode] = np.array(evidences)
    evidence_err_dict[recovery_mode] = np.array(evidence_errs)
    print(evidences)
    for k, v in evidence_dict.items():
        plt.plot(np.arange(len(v)), np.array(v)-ref_evidence, label=f"{label_dict_mean[mean_model]}",
                 linestyle=linestyle_dict[mean_model])

plt.xlabel("Number of flare components")
plt.ylabel(r"$\ln BF$")
plt.xticks(ticks=[1, 2, 3], labels=[1, 2, 3])
plt.legend(ncol=2)
# plt.ylim(-12, 8)
plt.tight_layout()
plt.savefig(f"results/Magnetar_mean_Ln_BF_plot.pdf")
plt.show()