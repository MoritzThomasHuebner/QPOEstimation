import matplotlib.pyplot as plt
import numpy as np

from QPOEstimation.result import GPResult
from QPOEstimation.utils import get_injection_outdir, get_injection_label

# import matplotlib.pyplot as plt
# plt.style.use('paper.mplstyle')

samples = []
outdir_qpo_qpo = get_injection_outdir(
    injection_mode='general_qpo', recovery_mode='general_qpo',
    likelihood_model='gaussian_process', base_injection_outdir='injections/injection_mss')
outdir_qpo_qpo = f"{outdir_qpo_qpo}/results"
outdir_qpo_red_noise = get_injection_outdir(
    injection_mode='general_qpo', recovery_mode='red_noise',
    likelihood_model='gaussian_process', base_injection_outdir='injections/injection_mss')
outdir_qpo_red_noise = f"{outdir_qpo_red_noise}/results"
outdir_red_noise_red_noise = get_injection_outdir(
    injection_mode='red_noise', recovery_mode='red_noise',
    likelihood_model='gaussian_process', base_injection_outdir='injections/injection_mss')
outdir_red_noise_red_noise = f"{outdir_red_noise_red_noise}/results"
outdir_red_noise_qpo = get_injection_outdir(
    injection_mode='red_noise', recovery_mode='general_qpo',
    likelihood_model='gaussian_process', base_injection_outdir='injections/injection_mss')
outdir_red_noise_qpo = f"{outdir_red_noise_qpo}/results"

ln_bfs_qpo_inj = []
ln_bfs_red_noise_inj = []

for injection_id in range(0, 1000):
    print(injection_id)
    label = get_injection_label(run_mode='entire_segment', injection_id=injection_id) + "_1_0s"
    try:
        res_qpo_qpo = GPResult.from_json(outdir=outdir_qpo_qpo, label=label)
        res_qpo_red_noise = GPResult.from_json(outdir=outdir_qpo_red_noise, label=label)
        res_red_noise_red_noise = GPResult.from_json(outdir=outdir_red_noise_red_noise, label=label)
        res_red_noise_qpo = GPResult.from_json(outdir=outdir_red_noise_qpo, label=label)
        ln_bfs_qpo_inj.append(res_qpo_qpo.log_evidence - res_qpo_red_noise.log_evidence)
        ln_bfs_red_noise_inj.append(res_red_noise_qpo.log_evidence - res_red_noise_red_noise.log_evidence)
    except (OSError, FileNotFoundError) as e:
        print(e)
        continue

np.savetxt('results/ln_bfs_qpo_inj_mss.txt', ln_bfs_qpo_inj)
np.savetxt('results/ln_bfs_red_noise_inj_mss.txt', ln_bfs_red_noise_inj)
bins = np.arange(-5, 25)
# bins = 'fd'
plt.hist(ln_bfs_qpo_inj, alpha=0.5, density=True, bins=bins, label="Simulated red noise plus QPO")
plt.hist(ln_bfs_red_noise_inj, alpha=0.5, density=True, bins=bins, label="Simulated red noise")
plt.semilogy()
plt.xlabel("$\ln BF_{\mathrm{QPO}}$")
plt.ylabel("$p(\ln BF_{\mathrm{QPO}})$")
plt.legend()
plt.tight_layout()
plt.savefig('results/mss_plot_new.pdf')
plt.show()

print(len(np.where(np.array(ln_bfs_qpo_inj) > 0)[0]))
print(len(np.where(np.array(ln_bfs_red_noise_inj) > 0)[0]))
