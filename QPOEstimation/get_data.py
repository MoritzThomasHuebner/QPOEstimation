import json
import numpy as np
from typing import Union

from .utils import get_injection_outdir, get_injection_label


def get_all_tte_magnetar_flare_data(
        magnetar_label: str, magnetar_tag: str, bin_size: float = 0.001, subtract_t0: bool = True,
        unbarycentred_time: bool = False, **kwargs) -> tuple:
    data = np.loadtxt(f"data/magnetar_flares/{magnetar_label}/{magnetar_tag}_data.txt")
    column = 1 if unbarycentred_time else 0
    ttes = data[:, column]

    counts, bin_edges = np.histogram(ttes, np.arange(ttes[0], ttes[-1], bin_size))
    times = np.array([bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2 for i in range(len(bin_edges) - 1)])
    if subtract_t0:
        times -= times[0]
    return times, counts


def get_tte_magnetar_flare_data_from_segment(
        start_time: float, end_time: float, magnetar_label: str, magnetar_tag: str, bin_size: float = 0.001,
        subtract_t0: bool = True, unbarycentred_time: bool = False, **kwargs) -> object:
    times, counts = get_all_tte_magnetar_flare_data(
        magnetar_label=magnetar_label, magnetar_tag=magnetar_tag, bin_size=bin_size,
        subtract_t0=subtract_t0, unbarycentred_time=unbarycentred_time, **kwargs)
    return truncate_data(times=times, counts=counts, start=start_time, stop=end_time)


def rebin(times: np.ndarray, counts: np.ndarray, rebin_factor: int) -> tuple:
    new_times = []
    new_counts = []
    for i in range(0, len(times), rebin_factor):
        if len(times) - i < rebin_factor:
            break
        c = 0
        for j in range(rebin_factor):
            c += counts[i + j]
        new_times.append(times[i])
        new_counts.append(c)
    return np.array(new_times), np.array(new_counts)


def get_all_binned_magnetar_flare_data(
        magnetar_label: str, magnetar_tag: str, subtract_t0: bool = True, rebin_factor: int = 1, **kwargs) -> tuple:
    data = np.loadtxt(f"data/magnetar_flares/{magnetar_label}/{magnetar_tag}_data.txt")
    times, counts = rebin(times=data[:, 0], counts=data[:, 1], rebin_factor=rebin_factor)
    if subtract_t0:
        times -= times[0]
    return times, counts


def get_all_binned_magnetar_flare_data_from_segment(
        start_time: float, end_time: float, magnetar_label: str, magnetar_tag: str,
        subtract_t0: bool = True, rebin_factor: int = 1, **kwargs) -> tuple:
    times, counts = get_all_binned_magnetar_flare_data(
        magnetar_label=magnetar_label, magnetar_tag=magnetar_tag,
        subtract_t0=subtract_t0, rebin_factor=rebin_factor, **kwargs)
    return truncate_data(times=times, counts=counts, start=start_time, stop=end_time)


def get_giant_flare_data(run_mode: str, **kwargs) -> tuple:
    """ Catch all function """
    return _GIANT_FLARE_RUN_MODES[run_mode](**kwargs)


def get_giant_flare_data_from_period(
        period_number: int = 0, run_id: int = 0, segment_step: float = 0.54,
        segment_length: float = 1, sampling_frequency: float = 256, **kwargs) -> tuple:
    pulse_period = 7.56  # see papers
    n_pulse_periods = 47
    time_offset = 20.0
    interpulse_periods = []
    for i in range(n_pulse_periods):
        interpulse_periods.append((time_offset + i * pulse_period, time_offset + (i + 1) * pulse_period))
    start = interpulse_periods[period_number][0] + run_id * segment_step
    stop = start + segment_length
    return get_giant_flare_data_from_segment(start_time=start, end_time=stop,
                                             sampling_frequency=sampling_frequency)


def get_giant_flare_data_from_segment(
        start_time: float = 10., end_time: float = 400.,
        sampling_frequency: float = 256, **kwargs) -> tuple:
    times, counts = get_all_giant_flare_data(sampling_frequency=sampling_frequency)
    return truncate_data(times=times, counts=counts, start=start_time, stop=end_time)


def get_all_giant_flare_data(sampling_frequency: float = 256, **kwargs) -> tuple:
    data = np.loadtxt(f"data/sgr1806_{sampling_frequency}Hz.dat")
    return data[:, 0], data[:, 1]


def truncate_data(times: np.ndarray, counts: np.ndarray, start: float, stop: float, yerr: np.ndarray = None) -> tuple:
    indices = np.where(np.logical_and(times > start, times < stop))[0]
    if yerr is None:
        return times[indices], counts[indices]
    else:
        return times[indices], counts[indices], yerr[indices]


def get_injection_data(
        injection_file_dir: str = "injection_files", injection_mode: str = "qpo", recovery_mode: str = "qpo",
        injection_likelihood_model: str = "celerite", injection_id: Union[float, np.ndarray] = 0,
        start_time: float = None, end_time: float = None, run_mode: str = "entire_segment", **kwargs) -> tuple:
    data = np.loadtxt(
        f"{injection_file_dir}/{injection_mode}/{injection_likelihood_model}/{str(injection_id).zfill(2)}_data.txt")
    if injection_mode == recovery_mode:
        with open(f"{injection_file_dir}/{injection_mode}/{injection_likelihood_model}/"
                  f"{str(injection_id).zfill(2)}_params.json", "r") as f:
            truths = json.load(f)
    else:
        truths = {}
    times = data[:, 0]
    counts = data[:, 1]
    try:
        yerr = data[:, 2]
    except Exception:
        yerr = None
    if run_mode == "select_time":
        try:
            times, counts, yerr = truncate_data(times, counts, start=start_time, stop=end_time, yerr=yerr)
        except ValueError:
            times, counts = truncate_data(times, counts, start=start_time, stop=end_time, yerr=yerr)
    return times, counts, yerr, truths


def get_grb_data_from_segment(
        grb_id: str, grb_binning: str, start_time: float, end_time: float, grb_detector: str = None,
        grb_energy_band: str = "all", grb_label: str = None, bin_size: str = None, **kwargs) -> tuple:
    times, y, yerr = get_all_grb_data(grb_binning=grb_binning, grb_id=grb_id, grb_detector=grb_detector,
                                      grb_label=grb_label, grb_energy_band=grb_energy_band, bin_size=bin_size)
    return truncate_data(times=times, counts=y, start=start_time, stop=end_time, yerr=yerr)


def get_all_grb_data(
        grb_id: str, grb_binning: str, grb_detector: str = None, grb_energy_band: str = "all", grb_label: str = None,
        bin_size: str = None, **kwargs) -> tuple:
    if grb_detector in ["swift", "konus"]:
        data_file = f"data/GRBs/GRB{grb_id}/{grb_binning}_lc_ascii_{grb_detector}.txt"
        data = np.loadtxt(data_file)
        times = data[:, 0]
        if grb_detector == "swift":
            if grb_energy_band == "15-25":
                y = data[:, 1]
                yerr = data[:, 2]
            elif grb_energy_band == "25-50":
                y = data[:, 3]
                yerr = data[:, 4]
            elif grb_energy_band == "50-100":
                y = data[:, 5]
                yerr = data[:, 6]
            elif grb_energy_band == "100-350":
                y = data[:, 7]
                yerr = data[:, 8]
            elif grb_energy_band in ["all", "15-350"]:
                y = data[:, 9]
                yerr = data[:, 10]
            else:
                raise ValueError(f"Energy band {grb_energy_band} not understood")
            return times, y, yerr
        elif grb_detector == "konus":
            y = data[:, 1]
            return times, y, np.sqrt(y)
    elif grb_detector == "batse":
        data_file = f"data/GRBs/GRB{grb_id}/GRB{grb_id}_{grb_energy_band}"
        data = np.loadtxt(data_file)
        times = data[:, 0]
        y = data[:, 1]
        yerr = np.sqrt(y)
        return times, y, yerr
    elif grb_detector.lower() == "asim":
        data_file = f"data/GRBs/GRB{grb_id}/{grb_label}.txt"
        ttes = np.loadtxt(data_file)
        y, bin_edges = np.histogram(ttes, np.arange(ttes[0], ttes[-1], bin_size))
        times = np.array([bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2 for i in range(len(bin_edges) - 1)])
        yerr = np.sqrt(y)
        return times, y, yerr


def get_grb_data(run_mode: str, **kwargs) -> tuple:
    """ Catch all function """
    return _GRB_RUN_MODES[run_mode](**kwargs)


def get_tte_magnetar_flare_data(run_mode: str, **kwargs) -> tuple:
    """ Catch all function """
    return _MAGNETAR_TTE_FLARE_RUN_MODES[run_mode](**kwargs)


def get_binned_magnetar_flare_data(run_mode: str, **kwargs) -> tuple:
    """ Catch all function """
    return _MAGNETAR_BINNED_FLARE_RUN_MODES[run_mode](**kwargs)


def get_solar_flare_data(run_mode: str, **kwargs) -> tuple:
    """ Catch all function """
    return _SOLAR_FLARE_RUN_MODES[run_mode](**kwargs)


def get_all_solar_flare_data(solar_flare_id: str = "go1520110128", solar_flare_folder: str = "goes", **kwargs) -> tuple:
    from astropy.io import fits
    data = fits.open(f"data/SolarFlare/{solar_flare_folder}/{solar_flare_id}.fits")

    times = data[2].data[0][0]
    flux = data[2].data[0][1][:, 0]
    # flux_err = data[2].data[0][1][:, 1]
    flux_err = np.zeros(len(times))
    return times, flux, flux_err


def get_solar_flare_data_from_segment(
        solar_flare_id: str = "go1520110128", solar_flare_folder: str = "goes",
        start_time: float = None, end_time: float = None, **kwargs) -> tuple:
    times, flux, flux_err = get_all_solar_flare_data(
        solar_flare_id=solar_flare_id, solar_flare_folder=solar_flare_folder)
    return truncate_data(times=times, counts=flux, start=start_time, stop=end_time, yerr=flux_err)


def get_hares_and_hounds_data(run_mode: str, **kwargs) -> tuple:
    """ Catch all function """
    return _HARES_AND_HOUNDS_RUN_MODES[run_mode](**kwargs)


def get_all_hares_and_hounds_data(
        hares_and_hounds_id: str = "5700", hares_and_hounds_round: str = "HH2", **kwargs) -> tuple:
    from astropy.io import fits
    data = fits.open(f"data/hares_and_hounds/{hares_and_hounds_round}/flare{hares_and_hounds_id}.fits")
    lc = data[1].data
    times = np.array([lc[i][0] for i in range(len(lc))])
    flux = np.array([lc[i][1] for i in range(len(lc))])
    return times, flux


def get_hares_and_hounds_data_from_segment(
        hares_and_hounds_id: str = "5700", hares_and_hounds_round: str = "HH2",
        start_time: float = None, end_time: float = None, **kwargs) -> tuple:
    times, flux = get_all_hares_and_hounds_data(hares_and_hounds_id=hares_and_hounds_id,
                                                hares_and_hounds_round=hares_and_hounds_round)
    return truncate_data(times=times, counts=flux, start=start_time, stop=end_time)


def get_hares_and_hounds_data_from_maximum(
        hares_and_hounds_id: str = "5700", hares_and_hounds_round: str = "HH2", **kwargs) -> tuple:
    times, flux = get_all_hares_and_hounds_data(hares_and_hounds_id=hares_and_hounds_id,
                                                hares_and_hounds_round=hares_and_hounds_round)
    max_index = np.argmax(flux)
    return truncate_data(times=times, counts=flux, start=times[max_index], stop=times[-1])


_GRB_RUN_MODES = dict(select_time=get_grb_data_from_segment,
                      entire_segment=get_all_grb_data)

_MAGNETAR_TTE_FLARE_RUN_MODES = dict(select_time=get_tte_magnetar_flare_data_from_segment,
                                     entire_segment=get_all_tte_magnetar_flare_data)

_MAGNETAR_BINNED_FLARE_RUN_MODES = dict(select_time=get_all_binned_magnetar_flare_data_from_segment,
                                        entire_segment=get_all_binned_magnetar_flare_data)

_SOLAR_FLARE_RUN_MODES = dict(select_time=get_solar_flare_data_from_segment,
                              entire_segment=get_all_solar_flare_data)

_GIANT_FLARE_RUN_MODES = dict(sliding_window=get_giant_flare_data_from_period,
                              select_time=get_giant_flare_data_from_segment,
                              entire_segment=get_all_giant_flare_data)

_HARES_AND_HOUNDS_RUN_MODES = dict(select_time=get_hares_and_hounds_data_from_segment,
                                   entire_segment=get_all_hares_and_hounds_data,
                                   from_maximum=get_hares_and_hounds_data_from_maximum)


def get_data(data_source: str, **kwargs) -> tuple:
    run_mode = kwargs["run_mode"]
    start_time = kwargs.get("start_time", 0)
    end_time = kwargs.get("end_time", 0)
    likelihood_model = kwargs.get("likelihood_model", None)
    recovery_mode_str = kwargs.get("recovery_mode_str", None)
    recovery_mode = kwargs.get("recovery_mode", None)
    if data_source == "giant_flare":
        times, y = get_giant_flare_data(**kwargs)
        yerr = np.sqrt(y)
        yerr[np.where(yerr == 0)[0]] = 1
        outdir = f"results/SGR_1806_20/{run_mode}/{kwargs['band']}/{recovery_mode_str}/{likelihood_model}/"
        if run_mode == "sliding_window":
            outdir += f"period_{kwargs['period_number']}/"
            label = f"{kwargs['run_id']}"
        elif run_mode == "select_time":
            label = f"{start_time}_{end_time}"
        elif run_mode == "entire_segment":
            label = "entire_segment"
        else:
            raise ValueError
    elif data_source == "magnetar_flare":
        times, y = get_tte_magnetar_flare_data(**kwargs)
        yerr = np.sqrt(y)
        yerr[np.where(yerr == 0)[0]] = 1
        outdir = f"results/magnetar_flares/{kwargs['magnetar_label']}/{kwargs['magnetar_tag']}/{run_mode}/" \
                 f"{recovery_mode_str}/{likelihood_model}/"
        if run_mode == "select_time":
            label = f"{start_time}_{end_time}"
        else:
            label = run_mode
    elif data_source == "magnetar_flare_binned":
        times, y = get_binned_magnetar_flare_data(**kwargs)
        yerr = np.sqrt(y)
        yerr[np.where(yerr == 0)[0]] = 1
        outdir = f"results/magnetar_flares/{kwargs['magnetar_label']}/{kwargs['magnetar_tag']}/{run_mode}/" \
                 f"{recovery_mode_str}/{likelihood_model}/"
        if run_mode == "select_time":
            label = f"{start_time}_{end_time}"
        else:
            label = run_mode
    elif data_source == "solar_flare":
        times, y, yerr = get_solar_flare_data(**kwargs)
        outdir = f"results/solar_flare_{kwargs['solar_flare_id']}/{run_mode}/{recovery_mode_str}/{likelihood_model}"
        if run_mode == "select_time":
            label = f"{start_time}_{end_time}"
        else:
            label = run_mode
    elif data_source == "grb":
        times, y, yerr = get_grb_data(**kwargs)
        outdir = f"results/GRB{kwargs['grb_id']}_{kwargs['grb_detector']}/" \
                 f"{run_mode}/{recovery_mode_str}/{likelihood_model}"
        if run_mode == "select_time":
            label = f"{start_time}_{end_time}"
        else:
            label = run_mode
        if kwargs["grb_energy_band"] != "all":
            label += f"_{kwargs['grb_energy_band']}keV"
        # times -= times[0]
    elif data_source == "injection":
        times, y, yerr, truths = get_injection_data(**kwargs)
        if yerr is None:
            yerr = np.zeros(len(y))
        outdir = get_injection_outdir(
            injection_mode=kwargs["injection_mode"], recovery_mode=recovery_mode,
            likelihood_model=kwargs["likelihood_model"], base_injection_outdir=kwargs["base_injection_outdir"])
        label = get_injection_label(run_mode, kwargs["injection_id"], start_time, end_time)
    elif data_source == "hares_and_hounds":
        times, y = get_hares_and_hounds_data(**kwargs)
        yerr = np.zeros(len(y))
        times -= times[0]
        outdir = f"results/hares_and_hounds_{kwargs['hares_and_hounds_round']}/" \
                 f"{kwargs['hares_and_hounds_id']}/{run_mode}/" \
                 f"{recovery_mode}/{likelihood_model}"
        if run_mode == "select_time":
            label = f"{start_time}_{end_time}"
        else:
            label = run_mode
    elif data_source == "test":
        data = np.loadtxt("data/test_goes_20130512_more.txt")
        times = data[:, 0]
        y = data[:, 1]
        yerr = data[:, 2]
        times, y, yerr = truncate_data(times=times, counts=y, yerr=yerr, start=start_time, stop=end_time)
        times -= times[0]

        outdir = f"results/goes_gp/{run_mode}/{recovery_mode}/{likelihood_model}"
        label = f"{start_time}_{end_time}"
    else:
        raise ValueError
    return times, y, yerr, outdir, label
