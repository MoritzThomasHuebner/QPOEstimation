import json
import numpy as np


def get_injection_data(injection_file_dir='injection_files', injection_mode='qpo', recovery_mode='qpo', likelihood_model='gaussian_process', injection_id=0):
    data = np.loadtxt(f'{injection_file_dir}/{injection_mode}/{likelihood_model}/{str(injection_id).zfill(2)}_data.txt')
    if injection_mode == recovery_mode:
        with open(f'{injection_file_dir}/{injection_mode}/{likelihood_model}/{str(injection_id).zfill(2)}_params.json', 'r') as f:
            truths = json.load(f)
    else:
        truths = {}
    return data[:, 0], data[:, 1], truths


def get_candidates_data(candidates_file_dir='candidates', band='5_64Hz', data_mode='normal', candidate_id=0, segment_length=1, sampling_frequency=256, alpha=0.02):
    candidates = np.loadtxt(f'{candidates_file_dir}/candidates_{band}_{data_mode}.txt')
    start = candidates[candidate_id][0]
    stop = start + segment_length
    return get_giant_flare_data_from_segment(start_time=start, end_time=stop, data_mode=data_mode, sampling_frequency=sampling_frequency, alpha=alpha)


def get_giant_flare_data_from_period(data_mode='normal', period_number=0, run_id=0, segment_step=0.54,
                                     segment_length=1, sampling_frequency=256, alpha=0.02):
    pulse_period = 7.56  # see papers
    n_pulse_periods = 47
    time_offset = 20.0
    interpulse_periods = []
    for i in range(n_pulse_periods):
        interpulse_periods.append((time_offset + i * pulse_period, time_offset + (i + 1) * pulse_period))
    start = interpulse_periods[period_number][0] + run_id * segment_step
    stop = start + segment_length
    return get_giant_flare_data_from_segment(start_time=start, end_time=stop, data_mode=data_mode, sampling_frequency=sampling_frequency, alpha=alpha)


def get_giant_flare_data_from_segment(start_time=10., end_time=400., data_mode='normal', sampling_frequency=256, alpha=0.02):
    data = get_all_giant_flare_data(data_mode=data_mode, sampling_frequency=sampling_frequency, alpha=alpha)
    return truncate_data(times=data[:, 0], counts=data[:, 1], start=start_time, stop=end_time)


def get_all_giant_flare_data(data_mode='normal', sampling_frequency=256, alpha=0.02):
    if data_mode == 'smoothed':
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_exp_smoothed_alpha_{alpha}.dat')
    elif data_mode == 'smoothed_residual':
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_exp_residual_alpha_{alpha}.dat')
    elif data_mode == 'blind_injection':
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz_{data_mode}.dat')
    else:
        data = np.loadtxt(f'data/sgr1806_{sampling_frequency}Hz.dat')
    return data


def truncate_data(times, counts, start, stop):
    indices = np.where(np.logical_and(times > start, times < stop))[0]
    return times[indices], counts[indices]

