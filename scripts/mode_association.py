import bilby
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.pyplot as plt
from QPOEstimation.prior.minimum import get_frequency_mode_priors

matplotlib.use('Qt5Agg')

# candidate_ids = [0, 1, 2, 3 10] #, 13]#, 14, 28, 33, 36]
candidate_ids = np.arange(0, 35)
kde_posteriors = []

kdes = []
band = '5_16Hz'
xs = np.linspace(5, 16, 3000)
for candidate_id in candidate_ids:
    print(candidate_id)
    res = bilby.result.read_in_result(f'sliding_window_{band}_candidates/one_qpo/results/{candidate_id}_result.json')
    res_no_qpo = bilby.result.read_in_result(f'sliding_window_{band}_candidates/no_qpo/results/{candidate_id}_result.json')
    log_bf = res.log_evidence - res_no_qpo.log_evidence
    if log_bf < 2:
        continue
    try:
        frequency_samples = np.loadtxt(f'sliding_window_{band}_candidates/one_qpo/results/{candidate_id}_frequency_samples.txt')
    except Exception:
        frequency_samples = []
        for i, sample in enumerate(res.posterior.iloc):
            frequency_samples.append(1 / np.exp(sample[f'kernel:log_P']))

        np.savetxt(f'sliding_window_{band}_candidates/one_qpo/results/{candidate_id}_frequency_samples.txt', frequency_samples)
    # if np.mean(frequency_samples) > 16:
    #     continue
    if frequency_samples[-1] > 14:
        continue
    if np.std(frequency_samples) > 1.5:
        continue
    kde = gaussian_kde(frequency_samples)


    plt.plot(xs, kde(xs), label=f"ID {candidate_id}, ln BF = {log_bf:.2f}")
    # plt.hist(frequency_samples, bins='fd', density=True, alpha=0.2)
    kdes.append(kde)
    kde_posteriors.append(bilby.core.prior.Interped(xx=xs, yy=kde(xs)))
plt.xlabel('frequency [Hz]')
plt.ylabel('PDF')
plt.xlim(5, 16)
plt.legend()
plt.savefig('all_posteriors_below_16Hz.pdf')
plt.show()

plt.plot(xs, np.sum([np.log(kde(xs)) for kde in kdes], axis=0))
plt.show()


class FrequencyModeLikelihood(bilby.core.likelihood.Likelihood):

    def __init__(self, n_freqs, posteriors):
        parameters = dict()
        self.posteriors = posteriors
        for i in range(n_freqs):
            parameters[f'log_f_{i}'] = 0
        super().__init__(parameters)

    def log_likelihood(self):
        p_unassociated = 1.
        for post in self.posteriors:
            for key, log_freq in self.parameters.items():
                freq = np.exp(log_freq)
                p_unassociated *= 1 - post.prob(freq)
        p_associated = (1 - p_unassociated)/len(self.parameters)
        return np.log(p_associated)

    # def log_likelihood(self):
    #     association_check = [False]*len(self.parameters)
    #     p_unassociated = 1.
    #     for post in self.posteriors:
    #         ps_unassociated = []
    #         for key, log_freq in self.parameters.items():
    #             freq = np.exp(log_freq)
    #             ps_unassociated.append(1 - post.prob(freq))
    #         minimum = np.amin(ps_unassociated)
    #         association_check[np.where(ps_unassociated == minimum)[0][0]] = True
    #         p_unassociated *= minimum
    #     for c in association_check:
    #         if not c:
    #             return -np.inf
    #     p_associated = 1 - p_unassociated
    #     return np.log(p_associated)


n_freqs = 2
likelihood = FrequencyModeLikelihood(n_freqs=n_freqs, posteriors=kde_posteriors)
priors = get_frequency_mode_priors(n_freqs=n_freqs, f_min=5, f_max=16, minimum_spacing=0.02)
print(likelihood.log_likelihood())

outdir = 'mode_association'
label = 'test_low_freq'


result = bilby.run_sampler(likelihood=likelihood, priors=priors,
                           outdir=outdir, label=label, sampler='dynesty',
                           nlive=400, resume=False, clean=True)
result.plot_corner()

max_like_freqs = []
max_like_params = result.posterior.iloc[-1]
for i in range(n_freqs):
    log_f = max_like_params[f'log_f_{i}']
    max_like_freqs.append(np.exp(log_f))
    plt.axvline(max_like_freqs[-1], color="black")

for kde in kdes:
    plt.plot(xs, kde(xs))
plt.xlabel('frequency [Hz]')
plt.ylabel('PDF')
plt.xlim(5, 16)
plt.legend()
plt.savefig('all_posteriors_below_16Hz.pdf')
plt.show()

print(max_like_freqs)