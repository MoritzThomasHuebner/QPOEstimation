import numpy as np
import bilby
from QPOEstimation.poisson import poisson_process
from QPOEstimation.model.series import sine_gaussian_with_background
import matplotlib.pyplot as plt
import matplotlib
import stingray
import json
matplotlib.use('Qt5Agg')


for i in range(100):
    priors = bilby.core.prior.PriorDict()
    priors['tau'] = bilby.core.prior.LogUniform(minimum=0.3, maximum=1.0)
    priors['offset'] = bilby.core.prior.LogUniform(minimum=1, maximum=50)
    priors['amplitude'] = bilby.core.prior.LogUniform(minimum=2, maximum=20)
    priors['mu'] = bilby.core.prior.Uniform(minimum=0.3, maximum=0.7)
    priors['sigma'] = bilby.core.prior.LogUniform(minimum=0.05, maximum=0.15)
    priors['frequency'] = bilby.core.prior.LogUniform(minimum=10, maximum=64)
    priors['phase'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi)

    params = priors.sample()
    print(params)

    times = np.linspace(0, 1, 256)
    poisson_counts = poisson_process(times, sine_gaussian_with_background, **params)
    np.savetxt(f'injection_files/{str(i).zfill(2)}_data.txt', np.array([times, poisson_counts]).T)
    with open(f'injection_files/{str(i).zfill(2)}_params.json', 'w') as f:
        json.dump(params, f)


# lc = stingray.Lightcurve(time=times, counts=poisson_counts)
# ps = stingray.Powerspectrum(lc=lc, norm='leahy')
# frequencies = ps.freq
# powers = ps.power
# plt.loglog(frequencies, powers)
# plt.show()
# plt.clf()


# likelihood = bilby.core.likelihood.PoissonLikelihood(x=times, y=poisson_counts, func=total_signal)
# result = bilby.run_sampler(likelihood=likelihood, priors=priors, label='test', outdir='testing_injection', sampler='dynesty', nlive=200, resume=False)
# result.plot_corner(truths=params)
#
# priors = bilby.core.prior.PriorDict()
# priors['tau'] = bilby.core.prior.LogUniform(minimum=0.3, maximum=1.0)
# priors['offset'] = bilby.core.prior.LogUniform(minimum=1, maximum=50)
#
# likelihood = bilby.core.likelihood.PoissonLikelihood(x=times, y=poisson_counts, func=background)
# result = bilby.run_sampler(likelihood=likelihood, priors=priors, label='test', outdir='testing_injection_background', sampler='dynesty', nlive=200, resume=False)
# result.plot_corner()
