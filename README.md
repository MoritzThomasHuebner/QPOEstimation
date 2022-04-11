# QPOEstimation

This is a small `python` package to analyse time series data for Quasi-periodic Oscillations (QPOs) using periodograms and Gaussian processes (GPs).
I implemented GP models using `celerite` and `george`, and use `bilby` for Bayesian inference.

My collaborators and I published the results on periodograms [here](https://ui.adsabs.harvard.edu/abs/2022ApJS..259...32H/abstract).
We also have a forthcoming publication on GPs.

Not all simulated data is available on the repository since it would take up too much space. I can provide the data upon request.

## Notes

Use `bilby` version 1.20 or higher to run this. If 1.20 has not been released yet, install from the feature branch on [https://git.ligo.org/lscsoft/bilby/-/merge_requests/1086](https://git.ligo.org/lscsoft/bilby/-/merge_requests/1086).

## Usage
The intended usage is shown in `analyse.py` and `analyse_periodogram.py`. `inject.py` shows how to create simulated data. 
Most of the keywords should be fairly self-explanatory. 
Most other scripts are for creating plots or post-processing of results.
I also provide a jupyter notebook in the `scripts` folder, which mostly follows `analyse.py`.
