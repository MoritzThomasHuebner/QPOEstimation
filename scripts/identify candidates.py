from collections import namedtuple
from itertools import groupby
from operator import itemgetter
import numpy as np

n_periods = 44


Candidate = namedtuple('Candidate', ['period_number', 'index_range', 'start', 'stop'])

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

candidates = []
pulse_period = 7.56  # see papers
for i in range(n_periods):
    log_bfs = np.loadtxt(f'sliding_window_fine/log_bfs_period_one_qpo_{i}')
    candidate_indices = np.where(log_bfs > 0.0)[0]
    rs = ranges(candidate_indices)
    for r in rs:
        if r[1] - r[0] >= 3:
            candidates.append(Candidate(
                period_number=i, index_range=(r[0], r[1]),
                start=r[0] * 0.05 + i * pulse_period,
                stop=r[0] * 0.05 + 0.2 + i * pulse_period))


starts = []
stops = []

for c in candidates:
    print(c)
    starts.append(c.start)
    stops.append(c.stop)

np.savetxt('candidates.txt', np.array([starts, stops]))
