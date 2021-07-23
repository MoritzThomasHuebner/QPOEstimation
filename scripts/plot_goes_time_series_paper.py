import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset

import QPOEstimation
from QPOEstimation.get_data import get_data
plt.style.use('paper.mplstyle')
# matplotlib.use('wxAgg')


times, y, _, outdir, label = get_data(
    data_source='solar_flare', run_mode='select_time', start_time=73000,
    end_time=75800,
    likelihood_model='whittle', solar_flare_folder='goes', solar_flare_id="go1520130512")
y = (y - np.mean(y)) / np.mean(y)

inset_indices = QPOEstimation.utils.get_indices_by_time(minimum_time=74700, maximum_time=74900, times=times)
inset_times = times[inset_indices]
inset_y = y[inset_indices]


fig, ax1 = plt.subplots()
ax1.plot(times, y)  # , label="Normalised flux [AU]")#'x', c='b', mew=2, alpha=0.8, label='Experiment')
ax1.set_xlabel(r'times [s]')
ax1.set_ylabel(r'Normalised flux [AU]')
ax1.set_title(r'GOES 1-8 $\mathrm{\AA}$')
# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
left = inset_times[0]
bottom = np.min(inset_y)
width = inset_times[-1] - inset_times[0]
height = np.max(inset_y) - np.min(inset_y)
ax2 = plt.axes([left, bottom, width, height])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.4, 0.1, 0.5, 0.5])
ax2.set_axes_locator(ip)
ax2.set_xticks([])
ax2.set_yticks([])
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=1, loc2=2, fc='none', ec='0.5', zorder=4)

ax2.plot(inset_times, inset_y)#, 'x', c='b', mew=2, alpha=0.8, label='Experiment')

# ax1.set_ylim(0, 26)
# ax2.set_yticks(np.arange(0, 2, 0.4))
# ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
# ax2.tick_params(axis='x', which='major', pad=8)
plt.tight_layout()
plt.savefig('paper_figures/goes_time_series.pdf')
plt.show()
