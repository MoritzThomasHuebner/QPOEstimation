import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import QPOEstimation
from QPOEstimation.get_data import *
from QPOEstimation.likelihood import get_kernel, get_mean_model, get_gp_likelihood
from QPOEstimation.parse import parse_args
from QPOEstimation.prior import get_priors
from QPOEstimation.prior.gp import *
from QPOEstimation.stabilisation import bar_lev
from QPOEstimation.utils import *

if len(sys.argv) > 1:
    parser = parse_args()
    args = parser.parse_args()

    data_source = args.data_source
    run_mode = args.run_mode
    sampling_frequency = args.sampling_frequency
    variance_stabilisation = boolean_string(args.variance_stabilisation)

    hares_and_hounds_id = args.hares_and_hounds_id
    hares_and_hounds_round = args.hares_and_hounds_round

    solar_flare_folder = args.solar_flare_folder
    solar_flare_id = args.solar_flare_id
    grb_id = args.grb_id
    grb_label = args.grb_label
    grb_binning = args.grb_binning
    grb_detector = args.grb_detector
    grb_energy_band = args.grb_energy_band
    magnetar_label = args.magnetar_label
    magnetar_tag = args.magnetar_tag
    bin_size = args.bin_size
    magnetar_subtract_t0 = boolean_string(args.magnetar_subtract_t0)
    magnetar_unbarycentred_time = boolean_string(args.magnetar_unbarycentred_time)

    rebin_factor = args.rebin_factor

    start_time = args.start_time
    end_time = args.end_time

    period_number = args.period_number
    run_id = args.run_id

    candidate_id = args.candidate_id

    polynomial_max = args.polynomial_max
    amplitude_min = args.amplitude_min
    amplitude_max = args.amplitude_max
    offset_min = args.offset_min
    offset_max = args.offset_max
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max
    t_0_min = args.t_0_min
    t_0_max = args.t_0_max
    offset = boolean_string(args.offset)

    min_log_a = args.min_log_a
    max_log_a = args.max_log_a
    min_log_c_red_noise = args.min_log_c_red_noise
    min_log_c_qpo = args.min_log_c_qpo
    max_log_c_red_noise = args.max_log_c_red_noise
    max_log_c_qpo = args.max_log_c_qpo
    minimum_window_spacing = args.minimum_window_spacing

    injection_id = args.injection_id
    injection_mode = args.injection_mode
    injection_file_dir = args.injection_file_dir
    injection_likelihood_model = args.injection_likelihood_model
    base_injection_outdir = args.base_injection_outdir
    normalisation = boolean_string(args.normalisation)

    recovery_mode = args.recovery_mode
    likelihood_model = args.likelihood_model
    background_model = args.background_model
    n_components = args.n_components
    jitter_term = boolean_string(args.jitter_term)

    band_minimum = args.band_minimum
    band_maximum = args.band_maximum

    segment_length = args.segment_length
    segment_step = args.segment_step

    nlive = args.nlive
    sample = args.sample
    use_ratio = boolean_string(args.use_ratio)

    try_load = boolean_string(args.try_load)
    resume = boolean_string(args.resume)
    plot = boolean_string(args.plot)
    suffix = args.suffix
else:
    # matplotlib.use("Qt5Agg")

    data_source = "giant_flare"  # "magnetar_flare_binned"
    run_mode = "select_time"
    sampling_frequency = 64
    variance_stabilisation = False

    hares_and_hounds_id = "455290"
    hares_and_hounds_round = "HH2"

    solar_flare_folder = "goes"
    solar_flare_id = "go1520130512"
    # solar_flare_id = "go1520110314"
    grb_id = "090709A"
    grb_label = ""
    grb_binning = "1s"
    grb_detector = "swift"
    grb_energy_band = "all"

    magnetar_label = "SGR_0501"
    magnetar_tag = "080823478_lcobs"
    bin_size = 0.001
    magnetar_subtract_t0 = True
    magnetar_unbarycentred_time = False
    rebin_factor = 8

    start_time = 101.060 + 20.0
    end_time = 103.060 + 20.0
    # start_time = 138.915 - 0.3 + 20.378
    # end_time = 139.915 + 0.5 + 20.378
    # start_time = 132.3 - 0.5 + 20.0
    # end_time = 132.3 + 0.5 + 20.0
    # start_time = -4.0
    # end_time = 103.0
    # start_time = 88.775 + 20.378
    # end_time = 90.775 + 20.378
    # start_time = -20.0
    # end_time = 20.0
    # start_time = 0 * 1e-3
    # end_time = 5 * 1e-3

    period_number = 14
    run_id = 6

    candidate_id = 5

    injection_id = 1
    base_injection_outdir = "injections/injection"

    offset = False
    polynomial_max = 2
    amplitude_min = None
    amplitude_max = None
    offset_min = None
    offset_max = None
    sigma_min = None
    sigma_max = None
    t_0_min = None
    t_0_max = None

    min_log_a = None
    max_log_a = None
    min_log_c_red_noise = None
    min_log_c_qpo = None
    max_log_c_red_noise = None
    max_log_c_qpo = None
    minimum_window_spacing = 0

    injection_mode = "qpo_plus_red_noise"
    injection_file_dir = "injection_files_pop"
    injection_likelihood_model = "whittle"
    recovery_mode = "qpo_plus_red_noise"
    likelihood_model = "celerite_windowed"
    background_model = "skew_exponential"
    n_components = 1
    jitter_term = False
    normalisation = False

    band_minimum = None
    band_maximum = None
    segment_length = 3.5
    # segment_step = 0.945  # Requires 8 steps
    segment_step = 0.23625  # Requires 32 steps

    sample = "rslice"
    nlive = 1000
    use_ratio = False

    try_load = True
    resume = True
    plot = True

if variance_stabilisation:
    suffix = f"_variance_stabilised"
else:
    suffix = ""
suffix += f"_{n_components}_{background_model}s"

mean_prior_bound_dict = dict(
    amplitude_min=amplitude_min,
    amplitude_max=amplitude_max,
    offset_min=offset_min,
    offset_max=offset_max,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    t_0_min=t_0_min,
    t_0_max=t_0_max
)

band = f"{band_minimum}_{band_maximum}Hz"

truths = None

recovery_mode_str = recovery_mode
if jitter_term:
    recovery_mode_str += "_jitter"

times, y, yerr, outdir, label = get_data(
    data_source=data_source, band=band, segment_length=segment_length,
    sampling_frequency=sampling_frequency,
    period_number=period_number, run_id=run_id, segment_step=segment_step, start_time=start_time, end_time=end_time,
    run_mode=run_mode, recovery_mode=recovery_mode, recovery_mode_str=recovery_mode_str, likelihood_model=likelihood_model,
    magnetar_label=magnetar_label,  magnetar_tag=magnetar_tag, bin_size=bin_size,
    magnetar_subtract_t0=magnetar_subtract_t0, magnetar_unbarycentred_time=magnetar_unbarycentred_time,
    rebin_factor=rebin_factor, solar_flare_folder=solar_flare_folder, solar_flare_id=solar_flare_id,
    grb_id=grb_id, grb_binning=grb_binning, grb_detector=grb_detector, grb_label=grb_label, grb_energy_band=grb_energy_band,
    injection_file_dir=injection_file_dir, injection_mode=injection_mode, injection_id=injection_id,
    injection_likelihood_model=injection_likelihood_model, hares_and_hounds_id=hares_and_hounds_id,
    hares_and_hounds_round=hares_and_hounds_round, base_injection_outdir=base_injection_outdir
    )


if normalisation:
    y = (y - np.min(y))/(np.max(y) - np.min(y)) * 1
    yerr = yerr/(np.max(y) - np.min(y))

if plot:
    # plt.errorbar(times, y, yerr=yerr, fmt=".k", capsize=0, label="data")
    # plt.errorbar(times, y, yerr=yerr, capsize=0, label="data")
    plt.step(times, y, label="flux")
    # plt.xlabel("time [s]")
    # plt.ylabel("counts/sec/det")
    # plt.title("GRB 090709A Swift-BAT 15-350 keV")
    plt.show()
    plt.clf()

    fs = 1/(times[1] - times[0])

    from scipy.signal import periodogram
    freqs, powers = periodogram(y, fs=fs, window="hann")
    plt.loglog()
    plt.step(freqs[1:], powers[1:])
    # plt.axvline(1/8.1, color="black", linestyle="--", label="QPO?")
    plt.xlabel("frequency [Hz]")
    plt.ylabel("Power [arb. units]")
    plt.legend()
    plt.show()


priors = get_priors(times=times, y=y, yerr=yerr, likelihood_model=likelihood_model, kernel_type=recovery_mode,
                    min_log_a=min_log_a, max_log_a=max_log_a,
                    min_log_c_red_noise=min_log_c_red_noise, max_log_c_red_noise=max_log_c_red_noise,
                    min_log_c_qpo=min_log_c_qpo, max_log_c_qpo=max_log_c_qpo, band_minimum=band_minimum, band_maximum=band_maximum,
                    model_type=background_model, polynomial_max=polynomial_max, minimum_spacing=minimum_window_spacing,
                    n_components=n_components, offset=offset, jitter_term=jitter_term, **mean_prior_bound_dict)

# priors["kernel:terms[0]:log_f"] = bilby.core.prior.Uniform(minimum=np.log(3000), maximum=8.517193191416348, name="kernel:terms[0]:log_f", latex_label="kernel:terms[0]:log_f", unit=None, boundary="reflective")
# suffix += "restricted_freq"

mean_model = get_mean_model(model_type=background_model, n_components=n_components, y=y, offset=offset,
                            likelihood_model=likelihood_model)
kernel = get_kernel(kernel_type=recovery_mode, jitter_term=jitter_term)
likelihood = get_gp_likelihood(mean_model=mean_model, kernel=kernel, times=times, y=y, yerr=yerr,
                               likelihood_model=likelihood_model)

meta_data = dict(kernel_type=recovery_mode, mean_model=background_model, times=times,
                 y=y, yerr=yerr, likelihood_model=likelihood_model, truths=truths, n_components=n_components,
                 offset=offset, jitter_term=jitter_term)


label += suffix
result = None
if try_load:
    try:
        result = QPOEstimation.result.GPResult.from_json(outdir=f"{outdir}/results", label=label)
        result.outdir = f"{outdir}/results"
    except IOError:
        bilby.utils.logger.info("No result file found. Starting from scratch")
if result is None:
    Path(f"{outdir}/results").mkdir(parents=True, exist_ok=True)
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                               label=label, sampler="dynesty", nlive=nlive, sample=sample,
                               resume=resume, use_ratio=use_ratio, result_class=QPOEstimation.result.GPResult,
                               meta_data=meta_data, save=True, gzip=False, nact=5)


if plot:
    result.plot_all(paper_style=True)


# clean up
for extension in ["_checkpoint_run.png", "_checkpoint_stats.png", "_checkpoint_trace.png", "_checkpoint_trace_unit.png",
                  "_dynesty.pickle", "_resume.pickle", "_result.json.old", "_samples.dat",
                  "_checkpoint_trace_unit.png"]:
    try:
        os.remove(f"{result.outdir}/{label}{extension}")
    except Exception:
        pass
