import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import argparse
import sys
import json

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--injection_mode", default="qpo", choices=["qpo", "white_noise", "red_noise"], type=str)
    parser.add_argument("--n_injections", default=100, type=int)
    parser.add_argument("--band_minimum", default=5, type=int)
    parser.add_argument("--band_maximum", default=64, type=int)
    args = parser.parse_args()

    injection_mode = args.injection_mode
    n_injections = args.n_injections
    band_minimum = args.band_minimum
    band_maximum = args.band_maximum
else:
    matplotlib.use('Qt5Agg')
    injection_mode = "red_noise"
    n_injections = 100
    band_minimum = 10
    band_maximum = 64

likelihood_model = 'gaussian_process'
injections = np.arange(0, n_injections)

band = f'{band_minimum}_{band_maximum}Hz'

log_bfs_one_qpo_gpr = []
log_bfs_one_qpo_whittle = []


# averaged runs

minimum_log_a = -2
maximum_log_a = 1
minimum_log_c = 1
maximum_log_c = 5
minimum_log_f = np.log(10)
maximum_log_f = np.log(64)

log_as = np.linspace(minimum_log_a, maximum_log_a, 10)
log_cs = np.linspace(minimum_log_c, maximum_log_c, 10)
if injection_mode == 'qpo':
    log_fs = np.linspace(minimum_log_f, maximum_log_f, 10)
else:
    log_fs = [0]

log_evidences_qpo = []
log_evidences_red_noise = []
averaged_log_bfs_qpo_v_red_noise = []
averaged_log_bfs_qpo_v_red_noise_err = []
for injection_id in range(100):
    log_a = log_as[int(str(injection_id).zfill(3)[1])]
    log_c = log_cs[int(str(injection_id).zfill(3)[2])]
    log_f = np.log(20)
    individual_log_bfs_qpo_v_red_noise = []
    for j in range(10):
        try:
            res_qpo = bilby.result.read_in_result(f"injection_{band}_normal_{injection_mode}/qpo/results/{str(j*100+injection_id).zfill(2)}_{likelihood_model}_result.json")
            res_red_noise = bilby.result.read_in_result(f"injection_{band}_normal_{injection_mode}/red_noise/results/{str(j*100+injection_id).zfill(2)}_{likelihood_model}_result.json")
            individual_log_bfs_qpo_v_red_noise.append((res_qpo.log_evidence - res_red_noise.log_evidence))
        except Exception as e:
            print(e)
    averaged_log_bfs_qpo_v_red_noise.append(np.mean(individual_log_bfs_qpo_v_red_noise))
    averaged_log_bfs_qpo_v_red_noise_err.append(np.std(individual_log_bfs_qpo_v_red_noise))

    print(averaged_log_bfs_qpo_v_red_noise[-1])
    print(injection_id)

for i in range(10):
    plt.errorbar(log_as, averaged_log_bfs_qpo_v_red_noise[i::10], label=f'ln c = {log_cs[i]:.2f}')
    plt.xlabel('ln a')
    plt.ylabel('ln BF')
plt.legend()
suffix = '20Hz'
plt.title("10 runs averaged")
if injection_mode == 'qpo':
    plt.savefig(f'ln_a_v_ln_BF_{injection_mode}_{suffix}.png')
else:
    plt.savefig(f'ln_a_v_ln_BF_{injection_mode}')
plt.show()
plt.clf()


for i in range(10):
    plt.semilogy(log_as, averaged_log_bfs_qpo_v_red_noise_err[i::10], label=f'ln c = {log_cs[i]:.2f}')
    plt.xlabel('ln a')
    plt.ylabel('$\Delta \ln \mathrm{BF}$')
plt.legend()
suffix = '20Hz'
plt.title("Standard deviation based on 10 injections")
if injection_mode == 'qpo':
    plt.savefig(f'ln_a_v_ln_BF_{injection_mode}_{suffix}_errs.png')
else:
    plt.savefig(f'ln_a_v_ln_BF_{injection_mode}_errs.png')
plt.show()
plt.clf()

for i in range(10):
    plt.plot(log_cs, averaged_log_bfs_qpo_v_red_noise[10 * i: 10 * i + 10], label=f'ln a = {log_as[i]:.2f}')
    plt.xlabel('ln c')
    plt.ylabel('ln BF')
plt.legend()
plt.title("10 runs averaged")
if injection_mode == 'qpo':
    plt.savefig(f'ln_c_v_ln_BF_{injection_mode}_{suffix}.png')
else:
    plt.savefig(f'ln_c_v_ln_BF_{injection_mode}')
plt.show()
plt.clf()

for i in range(10):
    plt.semilogy(log_cs, averaged_log_bfs_qpo_v_red_noise_err[10 * i: 10 * i + 10], label=f'ln a = {log_as[i]:.2f}')
    plt.xlabel('ln c')
    plt.ylabel('$\Delta \ln \mathrm{BF}$')
plt.legend()
plt.title("Standard deviation based on 10 injections")
if injection_mode == 'qpo':
    plt.savefig(f'ln_c_v_ln_BF_{injection_mode}_{suffix}_errs.png')
else:
    plt.savefig(f'ln_c_v_ln_BF_{injection_mode}_errs.png')

plt.show()
plt.clf()


log_bfs_qpo_red_noise_reshaped = np.reshape(averaged_log_bfs_qpo_v_red_noise, (10, 10))

cmap = matplotlib.cm.jet
ax = plt.contourf(log_as, log_cs, log_bfs_qpo_red_noise_reshaped,
                  cmap=cmap, levels=np.linspace(np.amin(log_bfs_qpo_red_noise_reshaped), np.amax(log_bfs_qpo_red_noise_reshaped), 1000))
plt.colorbar(ax)
plt.xlabel('ln c')
plt.ylabel('ln a')
plt.title("10 runs averaged")
if injection_mode == 'qpo':
    plt.savefig(f'ln_a_v_ln_c_v_ln_BF_{injection_mode}_{suffix}.png')
else:
    plt.savefig(f'ln_a_v_ln_c_v_ln_BF_{injection_mode}')
plt.show()
plt.clf()

# varied log f runs
# minimum_log_a = -2
# maximum_log_a = 1
# minimum_log_c = 1
# maximum_log_c = 4.8
# minimum_log_f = np.log(10)
# maximum_log_f = np.log(64)
#
# log_as = np.linspace(minimum_log_a, maximum_log_a, 10)
# log_cs = np.linspace(minimum_log_c, maximum_log_c, 10)
# if injection_mode == 'qpo':
#     log_fs = np.linspace(minimum_log_f, maximum_log_f, 10)
# else:
#     log_fs = [0]
#
# for j, log_f in enumerate(log_fs):
#     log_evidences_qpo = []
#     log_evidences_red_noise = []
#     log_bfs_one_qpo_red_noise = []
#     for injection_id in range(j*100, j*100+n_injections):
#         print(injection_id)
#
#         bilby.core.utils.logger.info(f"ID: {injection_id}")
#         log_f = log_fs[int(str(injection_id).zfill(3)[0])]
#         log_a = log_as[int(str(injection_id).zfill(3)[1])]
#         log_c = log_cs[int(str(injection_id).zfill(3)[2])]
#
#         try:
#             res_qpo = bilby.result.read_in_result(f"injection_{band}_{injection_mode}/qpo/results/{str(injection_id).zfill(2)}_{likelihood_model}_result.json")
#             res_red_noise = bilby.result.read_in_result(f"injection_{band}_{injection_mode}/red_noise/results/{str(injection_id).zfill(2)}_{likelihood_model}_result.json")
#             log_evidences_qpo.append(res_qpo.log_evidence)
#             log_evidences_red_noise.append(res_red_noise.log_evidence)
#             log_bfs_one_qpo_red_noise.append(res_qpo.log_evidence - res_red_noise.log_evidence)
#             print(log_bfs_one_qpo_red_noise[-1])
#
#         except Exception as e:
#             print(e)
#
#     for i in range(10):
#         plt.plot(log_as, log_bfs_one_qpo_red_noise[i::10], label=f'ln c = {log_cs[i]:.2f}')
#         plt.xlabel('ln a')
#         plt.ylabel('ln BF')
#     plt.legend()
#     suffix = f'{log_f:.2f}'
#     if injection_mode == 'qpo':
#         plt.savefig(f'ln_a_v_ln_BF_{injection_mode}_ln_f_{suffix}.png')
#     else:
#         plt.savefig(f'ln_a_v_ln_BF_{injection_mode}')
#     plt.show()
#     plt.clf()
#
#     for i in range(10):
#         plt.plot(log_cs, log_bfs_one_qpo_red_noise[10*i: 10*i+10], label=f'ln a = {log_as[i]:.2f}')
#         plt.xlabel('ln c')
#         plt.ylabel('ln BF')
#     plt.legend()
#     if injection_mode == 'qpo':
#         plt.savefig(f'ln_c_v_ln_BF_{injection_mode}_ln_f_{suffix}.png')
#     else:
#         plt.savefig(f'ln_c_v_ln_BF_{injection_mode}')
#
#     plt.show()
#     plt.clf()
#
#     log_bfs_one_qpo_red_noise_reshaped = np.reshape(log_bfs_one_qpo_red_noise, (10, 10))
#     plt.contourf(log_as, log_cs, log_bfs_one_qpo_red_noise_reshaped)
#     plt.colorbar()
#     plt.xlabel('ln c')
#     plt.ylabel('ln a')
#     if injection_mode == 'qpo':
#         plt.savefig(f'ln_a_v_ln_c_v_ln_BF_{injection_mode}_ln_f_{suffix}.png')
#     else:
#         plt.savefig(f'ln_a_v_ln_c_v_ln_BF_{injection_mode}')
#     plt.show()
#     plt.clf()

assert False

for injection_id in range(n_injections):
    try:
        res_no_qpo = bilby.result.read_in_result(f"sliding_window_{band}_injections/no_qpo/results/{str(injection_id).zfill(2)}_result.json")
        res_qpo = bilby.result.read_in_result(f"sliding_window_{band}_injections/one_qpo/results/{str(injection_id).zfill(2)}_result.json")
        res_no_qpo_whittle = bilby.result.read_in_result(f"sliding_window_{band}_injections/no_qpo/results/{str(injection_id).zfill(2)}_groth_result.json")
        res_one_qpo_whittle = bilby.result.read_in_result(f"sliding_window_{band}_injections/one_qpo/results/{str(injection_id).zfill(2)}_groth_result.json")
        # res_no_qpo_poisson = bilby.result.read_in_result(f"sliding_window_{band}_injections_fixed_log_b/no_qpo/results/{str(injection).zfill(2)}_poisson_result.json")
        # res_one_qpo_poisson = bilby.result.read_in_result(f"sliding_window_{band}_injections_fixed_log_b/one_qpo/results/{str(injection).zfill(2)}_poisson_result.json")
        # res_two_qpo = bilby.result.read_in_result(f"sliding_window_{band}_candidates/two_qpo/results/{candidate}_result.json")
        log_bfs_one_qpo_gpr.append(res_qpo.log_evidence - res_no_qpo.log_evidence)
        log_bfs_one_qpo_whittle.append(res_one_qpo_whittle.log_evidence - res_no_qpo_whittle.log_evidence)
        log_bfs_one_qpo_poisson.append(np.nan)
        # log_bfs_one_qpo_poisson.append(res_one_qpo_poisson.log_evidence - res_no_qpo_poisson.log_evidence)
        # log_bfs_two_qpo.append(res_two_qpo.log_evidence - res_no_qpo.log_evidence)
        # data = np.loadtxt(f"injection_files/{str(injection).zfill(2)}_data.txt")
        # plt.plot(data[:, 0], data[:, 1])
        # plt.xlabel('time [s]')
        # plt.ylabel('counts')
        # plt.savefig(f'injection_files/{str(injection).zfill(2)}_plot')
        # plt.clf()

        with open(f'injection_files/{str(injection_id).zfill(2)}_params.json', 'r') as f:
            injection_params = json.load(f)
            true_frequency = injection_params['frequency']

        try:
            log_P_samples = np.array(res_qpo.posterior['kernel:log_P'])
            frequency_samples_gpr = 1 / np.exp(log_P_samples)
        except Exception:
            log_f_samples = np.array(res_qpo.posterior['kernel:log_f'])
            frequency_samples_gpr = np.exp(log_f_samples)


        frequency_samples_whittle = np.array(res_one_qpo_whittle.posterior['central_frequency'])
        # frequency_samples_poisson = np.array(res_one_qpo_poisson.posterior['frequency'])

        perc_unc_gpr.append(np.std(frequency_samples_gpr)/np.mean(frequency_samples_gpr))
        perc_unc_whittle.append(np.std(frequency_samples_whittle)/np.mean(frequency_samples_whittle))
        # perc_unc_poisson.append(np.std(frequency_samples_poisson)/np.mean(frequency_samples_poisson))


        plt.hist(frequency_samples_gpr, bins="fd", density=True, alpha=0.3, label='gpr')
        plt.hist(frequency_samples_whittle, bins="fd", density=True, alpha=0.3, label='whittle')
        # plt.hist(frequency_samples_poisson, bins="fd", density=True, alpha=0.3, label='poisson')
        plt.axvline(true_frequency, label='injected frequency')
        plt.legend()
        plt.xlabel('frequency [Hz]')
        plt.ylabel('probability')
        plt.savefig(f"sliding_window_{band}_injections/frequency_posterior_{str(injection_id).zfill(2)}")
        plt.clf()

    except Exception as e:
        print(e)
        log_bfs_one_qpo_gpr.append(np.nan)
        log_bfs_one_qpo_whittle.append(np.nan)
        log_bfs_one_qpo_poisson.append(np.nan)
# np.savetxt(f'sliding_window_{band}_injections/log_bfs_one_qpo', np.array(log_bfs_one_qpo_gpr))
# np.savetxt(f'sliding_window_{band}_candidates/log_bfs_two_qpo', np.array(log_bfs_two_qpo))

print(f"ID:\tln BF GPR\tln BF Whittle\tln BF Poissson")
for i, log_bf_gpr, log_bf_whittle, log_bf_poisson in zip(np.arange(0, len(log_bfs_one_qpo_gpr)), log_bfs_one_qpo_gpr, log_bfs_one_qpo_whittle, log_bfs_one_qpo_poisson):
    print(f"{i}:\t{log_bf_gpr:.2f}\t{log_bf_whittle:.2f}\t{log_bf_poisson:.2f}")

print(f"Combined gpr ln BF: {np.sum(np.nan_to_num(log_bfs_one_qpo_gpr))}")
print(f"Combined whittle ln BF: {np.sum(np.nan_to_num(log_bfs_one_qpo_whittle))}")
print(f"Combined poisson ln BF: {np.sum(np.nan_to_num(log_bfs_one_qpo_poisson))}")


print(f"ID:\tGPR\tWhittle\tPoissson")
for i, gpr, whittle, poisson in zip(np.arange(0, len(perc_unc_gpr)), perc_unc_gpr, perc_unc_whittle, perc_unc_poisson):
    print(f"{i}:\t{gpr:.2f}\t{whittle:.2f}\t{poisson:.2f}")


print(f"Mean gpr rel. freq. err.: {np.mean(np.nan_to_num(perc_unc_gpr))}")
print(f"Mean whittle rel. freq. err.: {np.mean(np.nan_to_num(perc_unc_whittle))}")
print(f"Mean poisson rel. freq. err.: {np.mean(np.nan_to_num(perc_unc_poisson))}")
