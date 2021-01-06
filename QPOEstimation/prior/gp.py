import numpy as np

import bilby


def get_kernel_prior(kernel_type, min_log_a, max_log_a, min_log_c, band_minimum, band_maximum):
    priors = dict()
    if kernel_type == "white_noise":
        priors['kernel:log_sigma'] = bilby.core.prior.DeltaFunction(peak=-20, name='log_sigma')
    elif kernel_type == "qpo":
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='log_a')
        priors['kernel:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='log_b')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(band_maximum),
                                                          name='log_c')
        priors['kernel:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum),
                                                          maximum=np.log(band_maximum),
                                                          name='log_f')
        priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0,
                                                                 name='decay_constraint')
    elif kernel_type == "zeroed_qpo":
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='log_a')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(band_maximum),
                                                          name='log_c')
        priors['kernel:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum),
                                                          maximum=np.log(band_maximum),
                                                          name='log_f')
        priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0,
                                                                 name='decay_constraint')
    elif kernel_type == "red_noise":
        priors['kernel:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name='log_a')
        priors['kernel:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(band_maximum),
                                                          name='log_c')
    elif kernel_type == "mixed":
        priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a,
                                                                   name='terms[0]:log_a')
        priors['kernel:terms[0]:log_b'] = bilby.core.prior.DeltaFunction(peak=-10, name='terms[0]:log_b')
        priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(band_maximum),
                                                                   name='terms[0]:log_c')
        priors['kernel:terms[0]:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum),
                                                                   maximum=np.log(band_maximum),
                                                                   name='terms[0]:log_f')
        priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a,
                                                                   name='terms[1]:log_a')
        priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(band_maximum),
                                                                   name='terms[1]:log_c')
        priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0,
                                                                 name='decay_constraint')
    elif kernel_type == "zeroed_mixed":
        priors['kernel:terms[0]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a,
                                                                   name='terms[0]:log_a')
        priors['kernel:terms[0]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(band_maximum),
                                                                   name='terms[0]:log_c')
        priors['kernel:terms[0]:log_f'] = bilby.core.prior.Uniform(minimum=np.log(band_minimum),
                                                                   maximum=np.log(band_maximum),
                                                                   name='terms[0]:log_f')
        priors['kernel:terms[1]:log_a'] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a,
                                                                   name='terms[1]:log_a')
        priors['kernel:terms[1]:log_c'] = bilby.core.prior.Uniform(minimum=min_log_c, maximum=np.log(band_maximum),
                                                                   name='terms[1]:log_c')
        priors['decay_constraint'] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0,
                                                                 name='decay_constraint')
    else:
        raise ValueError('Recovery mode not defined')
    return priors