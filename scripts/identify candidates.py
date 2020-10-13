from collections import namedtuple
from itertools import groupby
from operator import itemgetter
import numpy as np

n_periods = 47
# band = '10_40Hz
# band = '5_16Hz'
band = '5_16Hz'
# band = 'below_16Hz'

Candidate = namedtuple('Candidate', ['period_number', 'index_range', 'start', 'stop', 'mean', 'std'])

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

candidates = []
pulse_period = 7.56  # see papers
time_offset = 20.0
segment_step = 0.27

for i in range(n_periods):
    log_bfs = np.loadtxt(f'sliding_window_{band}/log_bfs_period_one_qpo_{i}')
    mean_freqs = np.loadtxt(f'sliding_window_{band}/mean_frequencies_{i}')
    std_freqs = np.loadtxt(f'sliding_window_{band}/std_frequencies_{i}')
    candidate_indices = np.where(np.logical_and(log_bfs > 6, std_freqs/mean_freqs < 1/4))[0]
    # candidate_indices = np.where(log_bfs > 4)[0]
    rs = ranges(candidate_indices)

    for r in rs:
        if r[1] - r[0] >= 2:
            indexes = np.arange(r[0], r[1])
            preferred_index = np.where(log_bfs[indexes] == np.max(log_bfs[indexes]))[0][0] + indexes[0]
            print(log_bfs[preferred_index])
            # middle = np.int((r[1] + r[0])/2)
            candidates.append(Candidate(
                period_number=i, index_range=(r[0], r[1]),
                start=time_offset + i * pulse_period + preferred_index * segment_step,
                stop=time_offset + i * pulse_period + preferred_index * segment_step + 1,
                mean=mean_freqs[preferred_index], std=std_freqs[preferred_index]))


starts = []
stops = []

for c in candidates:
    print(c)
    starts.append(c.start)
    stops.append(c.stop)

np.savetxt(f'candidates/candidates_{band}.txt', np.array([starts, stops]).T)
