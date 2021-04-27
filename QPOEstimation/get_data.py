import json
import numpy as np


def get_all_tte_magnetar_flare_data(magnetar_label, tag, bin_size=0.001, subtract_t0=True,
                                    unbarycentred_time=False, **kwargs):
    data = np.loadtxt(f'data/magnetar_flares/{magnetar_label}/{tag}_data.txt')
    column = 1 if unbarycentred_time else 0
    ttes = data[:, column]

    counts, bin_edges = np.histogram(ttes, np.arange(ttes[0], ttes[-1], bin_size))
    times = np.array([bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2 for i in range(len(bin_edges) - 1)])
    if subtract_t0:
        times -= times[0]
    return times, counts


def get_tte_magnetar_flare_data_from_segment(start_time, end_time, magnetar_label, tag, bin_size=0.001,
                                             subtract_t0=True, unbarycentred_time=False, **kwargs):
    times, counts = get_all_tte_magnetar_flare_data(magnetar_label=magnetar_label, tag=tag, bin_size=bin_size, subtract_t0=subtract_t0,
                                                    unbarycentred_time=unbarycentred_time, **kwargs)
    return truncate_data(times=times, counts=counts, start=start_time, stop=end_time)


def get_giant_flare_data(run_mode, **kwargs):
    """ Catch all function """
    return _GIANT_FLARE_RUN_MODES[run_mode](**kwargs)


def get_candidates_data(candidates_file_dir='candidates', band='5_64Hz', data_mode='normal', candidate_id=0,
                        segment_length=1, sampling_frequency=256, alpha=0.02, **kwargs):
    candidates = np.loadtxt(f'{candidates_file_dir}/candidates_{band}_{data_mode}.txt')
    start = candidates[candidate_id][0]
    stop = start + segment_length
    return get_giant_flare_data_from_segment(start_time=start, end_time=stop, data_mode=data_mode,
                                             sampling_frequency=sampling_frequency, alpha=alpha)


def get_giant_flare_data_from_period(data_mode='normal', period_number=0, run_id=0, segment_step=0.54,
                                     segment_length=1, sampling_frequency=256, alpha=0.02, **kwargs):
    pulse_period = 7.56  # see papers
    n_pulse_periods = 47
    time_offset = 20.0
    interpulse_periods = []
    for i in range(n_pulse_periods):
        interpulse_periods.append((time_offset + i * pulse_period, time_offset + (i + 1) * pulse_period))
    start = interpulse_periods[period_number][0] + run_id * segment_step
    stop = start + segment_length
    return get_giant_flare_data_from_segment(start_time=start, end_time=stop, data_mode=data_mode,
                                             sampling_frequency=sampling_frequency, alpha=alpha)


def get_giant_flare_data_from_segment(start_time=10., end_time=400., data_mode='normal',
                                      sampling_frequency=256, alpha=0.02, **kwargs):
    times, counts = get_all_giant_flare_data(data_mode=data_mode, sampling_frequency=sampling_frequency, alpha=alpha)
    return truncate_data(times=times, counts=counts, start=start_time, stop=end_time)


def get_all_giant_flare_data(data_mode='normal', sampling_frequency=256, alpha=0.02, **kwargs):
    if data_mode == 'smoothed':
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_exp_smoothed_alpha_{alpha}.dat')
    elif data_mode == 'smoothed_residual':
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_exp_residual_alpha_{alpha}.dat')
    elif data_mode == 'blind_injection':
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_{data_mode}.dat')
    else:
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz.dat')
    return data[:, 0], data[:, 1]


def truncate_data(times, counts, start, stop, yerr=None):
    indices = np.where(np.logical_and(times > start, times < stop))[0]
    if yerr is None:
        return times[indices], counts[indices]
    else:
        return times[indices], counts[indices], yerr[indices]


def get_injection_data(injection_file_dir='injection_files', injection_mode='qpo', recovery_mode='qpo',
                       likelihood_model='gaussian_process', injection_id=0, **kwargs):
    data = np.loadtxt(f'{injection_file_dir}/{injection_mode}/{likelihood_model}/{str(injection_id).zfill(2)}_data.txt')
    if injection_mode == recovery_mode:
        with open(f'{injection_file_dir}/{injection_mode}/{likelihood_model}/'
                  f'{str(injection_id).zfill(2)}_params.json', 'r') as f:
            truths = json.load(f)
    else:
        truths = {}
    return data[:, 0], data[:, 1], truths


def get_grb_data_from_segment(
        grb_id, grb_binning, start_time, end_time, grb_detector=None, **kwargs):
    times, y, yerr = get_all_grb_data(grb_binning=grb_binning, grb_id=grb_id, grb_detector=grb_detector)
    return truncate_data(times=times, counts=y, start=start_time, stop=end_time, yerr=yerr)


def get_all_grb_data(grb_id, grb_binning, grb_detector=None, **kwargs):
    data_file = f'data/GRB{grb_id}/{grb_binning}_lc_ascii_{grb_detector}.txt'
    data = np.loadtxt(data_file)
    times = data[:, 0]
    if grb_detector == 'swift':
        y = data[:, 9]
        yerr = data[:, 10]
        return times, y, yerr
    elif grb_detector == 'konus':
        y = data[:, 1]
        return times, y, np.sqrt(y)


def get_grb_data(run_mode, **kwargs):
    """ Catch all function """
    return _GRB_RUN_MODES[run_mode](**kwargs)


def get_tte_magnetar_flare_data(run_mode, **kwargs):
    """ Catch all function """
    return _MAGNETAR_FLARE_RUN_MODES[run_mode](**kwargs)


def get_solar_flare_data(run_mode, **kwargs):
    """ Catch all function """
    return _SOLAR_FLARE_RUN_MODES[run_mode](**kwargs)


def get_all_solar_flare_data(solar_flare_id="go1520110128", **kwargs):
    from astropy.io import fits
    data = fits.open(f'data/SolarFlare/{solar_flare_id}.fits')

    times = data[2].data[0][0]
    flux = data[2].data[0][1][:, 0]
    flux_err = data[2].data[0][1][:, 1]
    return times, flux, flux_err



def get_solar_flare_data_from_segment(solar_flare_id="go1520110128", start_time=None, end_time=None, **kwargs):
    times, flux, flux_err = get_all_solar_flare_data(solar_flare_id=solar_flare_id)
    return truncate_data(times=times, counts=flux, start=start_time, stop=end_time, yerr=flux_err)


_GRB_RUN_MODES = dict(select_time=get_grb_data_from_segment,
                      entire_segment=get_all_grb_data)

_MAGNETAR_FLARE_RUN_MODES = dict(select_time=get_tte_magnetar_flare_data_from_segment,
                                 entire_segment=get_all_tte_magnetar_flare_data)

_SOLAR_FLARE_RUN_MODES = dict(select_time=get_solar_flare_data_from_segment,
                              entire_segment=get_all_solar_flare_data)

_GIANT_FLARE_RUN_MODES = dict(candidates=get_candidates_data,
                              sliding_window=get_giant_flare_data_from_period,
                              select_time=get_giant_flare_data_from_segment,
                              entire_segment=get_all_giant_flare_data)
