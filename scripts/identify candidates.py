from collections import namedtuple
from itertools import groupby
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

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
import bilby

Candidate = namedtuple('Candidate', ['period_number', 'index_range', 'start', 'stop',
                                     'mean', 'std', 'log_bf', 'rotational_phase'])

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
    candidate_indices = np.where(log_bfs > 6)[0]
    # candidate_indices = np.where(log_bfs > 4)[0]
    # candidate_indices = np.where(log_bfs > 4)[0]
    rs = ranges(candidate_indices)

    for r in rs:
        if r[1] == r[0]:
            indexes = [r[0]]
        else:
            indexes = np.arange(r[0], r[1])
        preferred_index = np.where(log_bfs[indexes] == np.max(log_bfs[indexes]))[0][0] + indexes[0]
        rotational_phase = 2*np.pi * preferred_index/len(log_bfs)
        candidates.append(Candidate(
            period_number=i, index_range=(r[0], r[1]),
            start=time_offset + i * pulse_period + preferred_index * segment_step,
            stop=time_offset + i * pulse_period + preferred_index * segment_step + 1,
            mean=mean_freqs[preferred_index], std=std_freqs[preferred_index], log_bf=log_bfs[preferred_index],
            rotational_phase=rotational_phase))


starts = []
stops = []

for c in candidates:
    print(c)
    starts.append(c.start)
    stops.append(c.stop)

np.savetxt(f'candidates/candidates_{output_band}.txt', np.array([starts, stops]).T)

# means = []
# stds = []
suffix = ""


period_numbers = [candidate.period_number for candidate in candidates]
rotational_phase = [candidate.rotational_phase for candidate in candidates]
means = [candidate.mean for candidate in candidates]
stds = [candidate.std for candidate in candidates]

plt.errorbar(period_numbers, means, yerr=stds, capsize=5, fmt=".k")
plt.xlabel("Period Number")
plt.ylabel("Frequency")
plt.title("QPOs with ln BF > 6")
plt.savefig(f"Frequencies_vs_periods_{output_band}{suffix}.png")
plt.show()
plt.clf()

data = np.loadtxt('data/sgr1806_64Hz.dat')
times = data[:, 0]
counts = data[:, 1]

pulse_period = 7.56  # see papers
start = 20
segment_length = 7.56
stop = start + segment_length

indices = np.where(np.logical_and(times > start, times < stop))
t = times[indices]
t -= t[0]
t /= t[-1]
t *= 2*np.pi
c = counts[indices]
plt.plot(t, c, label='data')
plt.xlabel("Rotational Phase")
plt.ylabel("Counts")
plt.savefig(f"Counts_vs_phase.png")
plt.show()
plt.clf()


plt.errorbar(rotational_phase, means, yerr=stds, capsize=5, fmt=".k")
plt.xlim(0, 2*np.pi)
plt.xlabel("Rotational Phase")
plt.ylabel("Frequency")
plt.title("QPOs with ln BF > 6")
plt.savefig(f"Frequencies_vs_phase_{output_band}{suffix}.png")
plt.show()
plt.clf()