import bilby
import numpy as np


class SlabSpikePrior(bilby.core.prior.Prior):

    def __init__(self, name=None, latex_label=None, unit=None, minimum=0.,
                 maximum=1., spike_height=0.5, spike_loc=None, check_range_nonzero=True, boundary=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit, minimum=minimum,
                         maximum=maximum, check_range_nonzero=check_range_nonzero, boundary=boundary)
        if spike_loc is None:
            self.spike_loc = minimum
        else:
            self.spike_loc = spike_loc
        self.spike_height = spike_height

    @property
    def segment_length(self):
        return self.maximum - self.minimum

    def rescale(self, val):
        val = np.atleast_1d(val)
        res = np.zeros(len(val))
        non_spike_frac = 1 - self.spike_height
        frac_below_spike = (self.spike_loc - self.minimum)/self.segment_length * non_spike_frac
        spike_start = frac_below_spike
        lower_indices = np.where(val < spike_start)
        intermediate_indices = np.where(np.logical_and(val >= spike_start, val <= spike_start + self.spike_height))
        higher_indices = np.where(val > spike_start + self.spike_height)
        res[lower_indices] = val[lower_indices] * self.segment_length / non_spike_frac + self.minimum
        res[intermediate_indices] = spike_start * self.segment_length / non_spike_frac + self.minimum
        res[higher_indices] = (val[higher_indices] - self.spike_height) * self.segment_length / non_spike_frac + self.minimum
        return res

    def prob(self, val):
        return ((val >= self.minimum) & (val <= self.maximum)) * (self.spike_height + (1 - self.spike_height) / (self.maximum - self.minimum))

    def ln_prob(self, val):
        return np.log(self.prob(val))


ConditionalSlabSpikePrior = bilby.core.prior.conditional_prior_factory(SlabSpikePrior)