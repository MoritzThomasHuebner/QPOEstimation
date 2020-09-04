import bilby
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

# candidate_ids = [0, 1, 2]
candidate_ids = np.arange(0, 89)
kde_posteriors = []

for candidate_id in candidate_ids:
    print(candidate_id)
    res = bilby.result.read_in_result(f'sliding_window_candidates/one_qpo/{candidate_id}_result.json')

    try:
        frequency_samples = np.loadtxt(f'sliding_window_candidates/one_qpo/{candidate_id}_frequency_samples.txt')
    except Exception:
        frequency_samples = []
        for i, sample in enumerate(res.posterior.iloc):
            frequency_samples.append(1 / np.exp(sample[f'kernel:terms[1]:log_P']))

        np.savetxt(f'sliding_window_candidates/one_qpo/{candidate_id}_frequency_samples.txt', frequency_samples)
    # if frequency_samples[-1] > 40:
    #     continue
    kde = gaussian_kde(frequency_samples)

    xs = np.linspace(5, 128, 3000)
    plt.plot(xs, kde(xs), label="From posterior")
    # plt.hist(frequency_samples, bins='fd', density=True, alpha=0.2)

    kde_posteriors.append(bilby.core.prior.Interped(xx=xs, yy=kde(xs)))
plt.xlabel('frequency [Hz]')
plt.ylabel('PDF')
plt.savefig('all_posteriors.pdf')
plt.show()


class AssociationLikelihood(bilby.core.likelihood.Likelihood):

    def __init__(self, posteriors):
        super().__init__(parameters=dict(log_f_0=0))
        self.posteriors = posteriors

    def log_likelihood(self):
        return np.sum(np.log(self.p_associated_any()))

    def p_associated_any(self):
        ps = []
        for prob in self.posteriors:
            ps.append(1 - self.p_unassociated(prob))
        return ps

    def p_unassociated(self, prob):
        p = 1
        ll = 2
        frequency = 0
        while frequency < prob.maximum:
            frequency = self.frequency_at_mode(ll=ll)
            if prob.minimum <= frequency:
                p_associated = prob.prob(frequency) * 1/(ll+1)  # 1/l correction factor?
                if ll == 1:
                    p_associated = 0
                p *= 1 - p_associated
            ll += 1
        return p

    def frequency_at_mode(self, ll):
        return np.exp(self.parameters['log_f_0']) * np.sqrt(ll*(ll+1))


likelihood = AssociationLikelihood(posteriors=kde_posteriors)
priors = bilby.core.prior.PriorDict()
priors['log_f_0'] = bilby.core.prior.Uniform(minimum=-7, maximum=2.3, name='log_f_0')
p = bilby.core.prior.Uniform(minimum=-3, maximum=2.3, name='log_f_0')

outdir = 'mode_association'
label = 'test_low_freq'


log_fs = np.linspace(-3, 2.3, 2000)
post = []
for log_f in log_fs:
    likelihood.parameters['log_f_0'] = log_f
    post.append(np.exp(likelihood.log_likelihood() + p.ln_prob(log_f)))


plt.plot(np.exp(log_fs), post)
plt.show()

# fs = np.linspace(5, 40, 2000)
#
#
# posterior_prob = []
# for f in fs:
#     likelihood.parameters = dict(f_0=f)
#     posterior_prob.append(likelihood.log_likelihood() + priors['f_0'].ln_prob(f))
#     print(posterior_prob[-1])
# plt.plot(fs, posterior_prob)
# plt.show()



# result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=outdir, label=label, sampler='dynesty',
#                            nlive=100, resume=False, clean=True)
# result.plot_corner()
