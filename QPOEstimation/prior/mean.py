import bilby


def get_polynomial_prior(polynomial_max=10, order=4):
    priors = bilby.core.prior.PriorDict()
    for i in range(order + 1):
        if polynomial_max == 0:
            priors[f'mean:a{i}'] = 0
        else:
            priors[f'mean:a{i}'] = bilby.core.prior.Uniform(
                minimum=-polynomial_max, maximum=polynomial_max, name=f'mean:a{i}')
    return priors
