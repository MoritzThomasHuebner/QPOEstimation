from collections import namedtuple
from itertools import groupby
from operator import itemgetter
import numpy as np

n_periods = 47
# band_minimum = 64
# band_maximum = 128
band_minimum = 5
band_maximum = 64
# band = '10_40Hz
# band = '5_16Hz'
output_band = f'{band_minimum}_{band_maximum}Hz'
search_band = f'{band_minimum}_{band_maximum}Hz'
# band = 'below_16Hz'

Candidate = namedtuple('Candidate', ['period_number', 'index_range', 'start', 'stop', 'mean', 'std', 'log_bf'])

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
    log_bfs = np.loadtxt(f'sliding_window_{search_band}/log_bfs_period_one_qpo_{i}')
    mean_freqs = np.loadtxt(f'sliding_window_{search_band}/mean_frequencies_{i}')
    std_freqs = np.loadtxt(f'sliding_window_{search_band}/std_frequencies_{i}')
    # candidate_indices = np.where(np.logical_and(log_bfs > 4, std_freqs/mean_freqs < 1/4))[0]
    candidate_indices = np.where(log_bfs > 5)[0]
    # candidate_indices = np.where(log_bfs > 4)[0]
    # candidate_indices = np.where(log_bfs > 4)[0]
    rs = ranges(candidate_indices)

    for r in rs:
        if r[1] == r[0]:
            indexes = [r[0]]
        else:
            indexes = np.arange(r[0], r[1])
        preferred_index = np.where(log_bfs[indexes] == np.max(log_bfs[indexes]))[0][0] + indexes[0]
        candidates.append(Candidate(
            period_number=i, index_range=(r[0], r[1]),
            start=time_offset + i * pulse_period + preferred_index * segment_step,
            stop=time_offset + i * pulse_period + preferred_index * segment_step + 1,
            mean=mean_freqs[preferred_index], std=std_freqs[preferred_index], log_bf=log_bfs[preferred_index]))


starts = []
stops = []

for c in candidates:
    print(c)
    starts.append(c.start)
    stops.append(c.stop)

np.savetxt(f'candidates/candidates_{output_band}.txt', np.array([starts, stops]).T)
