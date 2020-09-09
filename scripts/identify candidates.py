from collections import namedtuple
from itertools import groupby
from operator import itemgetter
import numpy as np

n_periods = 47
band = '16_32Hz'

Candidate = namedtuple('Candidate', ['period_number', 'index_range', 'start', 'stop'])

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

candidates = []
pulse_period = 7.56  # see papers
offset = 10.0
for i in range(n_periods):
    log_bfs = np.loadtxt(f'sliding_window_{band}/log_bfs_period_one_qpo_{i}')
    candidate_indices = np.where(log_bfs > 2.0)[0]
    rs = ranges(candidate_indices)
    for r in rs:
        if r[1] - r[0] >= 3:
            candidates.append(Candidate(
                period_number=i, index_range=(r[0], r[1]),
                start=offset + i * pulse_period + r[0] * 0.135 + 0.2, # + 0.5,
                stop=offset + i * pulse_period + r[1] * 0.135 + 1 - 0.2)) # ))


starts = []
stops = []

for c in candidates:
    print(c)
    starts.append(c.start)
    stops.append(c.stop)

np.savetxt(f'candidates_{band}.txt', np.array([starts, stops]).T)
