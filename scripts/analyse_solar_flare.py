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

linestyle_dict = dict(polynomial="solid")
label_dict_mean = dict(polynomial="polymial")
label_dict_kernel = dict(qpo_plus_red_noise="qpo+rn", red_noise="rn")

plt.figure(figsize=(9.2, 7.2))
# plt.figure(dpi=150)

mean_model = "polynomial"

for recovery_mode in ["qpo_plus_red_noise", "red_noise"]:
    evidences = []
    evidence_errs = []
    res_list = []
    for n_component in range(2, 5):
        try:
            res = GPResult.from_json(
                outdir=f"results/solar_flare_go1520130512/select_time/{recovery_mode}/"
                       f"celerite/results/",
                label=f"74700.0_74900.0_{n_component}_{mean_model}s")
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
    color_dict = dict(qpo_plus_red_noise="blue", red_noise="red", double_qpo="green")
    plt.plot(np.arange(len(v)), np.array(v), label=f"{label_dict_mean[mean_model]},  {label_dict_kernel[k]}",
             color=color_dict[k], linestyle=linestyle_dict[mean_model])

plt.xlabel("Polynomial order")
plt.ylabel(f"ln Z")
plt.xticks(ticks=[0, 1, 2], labels=[1, 2, 3])
# plt.ylim(88, 104)

# plt.ylim(-372, -357)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"results/Solar_Ln_Z_plot.pdf")
plt.show()

# plt.figure(figsize=(9.2, 7.2))
# for mean_model, res_list in res_dict.items():
#     for i, res in enumerate(res_list):
#         n_component = 1 + i
#         log_a_samples = np.array(res.posterior["kernel:terms[0]:log_a"])
#         log_c_samples = np.array(res.posterior["kernel:terms[0]:log_c"])
#         log_f_samples = np.array(res.posterior["kernel:terms[0]:log_f"])
#         power_samples = np.log(
#             power_qpo(a=np.exp(log_a_samples), c=np.exp(log_c_samples), f=np.exp(log_f_samples)))
#         plt.hist(power_samples, bins="fd", density=True, histtype="step",
#                  label=f"{n_component} {label_dict_mean[mean_model]} flares")
# plt.xlabel("$\ln P_{\mathrm{QPO}}$")
# plt.ylabel("Normalised PDF")
# plt.legend()
# plt.savefig(f"Magnetar_qpo_power_hist.png")
# plt.clf()
