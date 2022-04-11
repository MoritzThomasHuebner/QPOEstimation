import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import periodogram

import QPOEstimation
from QPOEstimation.get_data import *
from QPOEstimation.parse import parse_args
from QPOEstimation.prior.psd import *
from QPOEstimation.utils import *

if len(sys.argv) > 1:
    # plt.style.use("paper.mplstyle")
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
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max
    t_0_min = args.t_0_min
    t_0_max = args.t_0_max

    minimum_window_spacing = args.minimum_window_spacing

    injection_id = args.injection_id
    injection_mode = args.injection_mode
    injection_file_dir = args.injection_file_dir
    injection_likelihood_model = args.injection_likelihood_model
    base_injection_outdir = args.base_injection_outdir

    recovery_mode = args.recovery_mode
    likelihood_model = args.likelihood_model
    normalisation = boolean_string(args.normalisation)

    window = args.window
    frequency_mask_minimum = args.frequency_mask_minimum
    frequency_mask_maximum = args.frequency_mask_maximum

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
else:
    matplotlib.use("Qt5Agg")

    data_source = "grb"
    run_mode = "select_time"
    sampling_frequency = 4096
    variance_stabilisation = False

    hares_and_hounds_id = "612579"
    hares_and_hounds_round = "HH2"

    solar_flare_folder = "goes"
    solar_flare_id = "go1520130512"
    grb_id = "200415A"
    grb_label = "ASIM_CLEANED_LED"
    grb_binning = "64ms"
    grb_detector = "asim"
    grb_energy_band = "all"

    magnetar_label = "SGR_0501"
    magnetar_tag = "080823478_lcobs"
    # bin_size = 100*1e-6
    bin_size = 0.1*1e-3
    magnetar_subtract_t0 = True
    magnetar_unbarycentred_time = False
    rebin_factor = 1

    start_time = 0 * 1e-3
    end_time = 100 * 1e-3
    # start_time = 73020
    # end_time = 75780
    # start_time = 210.735 + 20.378 + 0.5
    # end_time = 210.735 + 20.378 + 1

    period_number = 14
    run_id = 6

    candidate_id = 5

    injection_id = 1

    polynomial_max = 1000000
    amplitude_min = None
    amplitude_max = None
    # sigma_min = 0.1
    # sigma_max = 10000
    sigma_min = 2
    sigma_max = 2
    # t_0_min = 1e-3
    # t_0_max = 1000
    t_0_min = None
    t_0_max = None

    minimum_window_spacing = 0

    injection_mode = "qpo_plus_red_noise"
    injection_file_dir = "injection_files_pop"
    injection_likelihood_model = "whittle"
    base_injection_outdir = "injections/injection"

    recovery_mode = "red_noise"
    likelihood_model = "whittle"
    normalisation = True

    window = "tukey"
    frequency_mask_minimum = None
    frequency_mask_maximum = None

    band_minimum = 500
    band_maximum = 10000
    segment_length = 3.5
    # segment_step = 0.945  # Requires 8 steps
    segment_step = 0.23625  # Requires 32 steps

    sample = "rwalk"
    nlive = 500
    use_ratio = False

    try_load = True
    resume = True
    plot = True

band = f"{band_minimum}_{band_maximum}Hz"

truths = None

recovery_mode_str = recovery_mode

times, y, _, outdir, label = get_data(
    data_source=data_source, band=band, segment_length=segment_length,
    sampling_frequency=sampling_frequency,
    period_number=period_number, run_id=run_id, segment_step=segment_step, start_time=start_time, end_time=end_time,
    run_mode=run_mode, recovery_mode=recovery_mode, recovery_mode_str=recovery_mode_str,
    likelihood_model=likelihood_model, magnetar_label=magnetar_label, magnetar_tag=magnetar_tag,
    bin_size=bin_size, magnetar_subtract_t0=magnetar_subtract_t0,
    magnetar_unbarycentred_time=magnetar_unbarycentred_time, rebin_factor=rebin_factor,
    solar_flare_folder=solar_flare_folder, solar_flare_id=solar_flare_id, grb_id=grb_id, grb_binning=grb_binning,
    grb_detector=grb_detector, grb_energy_band=grb_energy_band, grb_label=grb_label,
    injection_file_dir=injection_file_dir, injection_mode=injection_mode, injection_id=injection_id,
    injection_likelihood_model=injection_likelihood_model, hares_and_hounds_id=hares_and_hounds_id,
    hares_and_hounds_round=hares_and_hounds_round, base_injection_outdir=base_injection_outdir
    )

if grb_label == "ASIM_CLEANED_LED":
    outdir += "LED"
elif grb_label == "ASIM_CLEANED_HED":
    outdir += "HED"

sampling_frequency = 1 / (times[1] - times[0])
if window == "tukey":
    window = ("tukey", 0.05)


if normalisation:
    # y = (y - np.mean(y))/np.mean(y)
    from stingray.lightcurve import Lightcurve
    from stingray.powerspectrum import Powerspectrum
    from scipy.signal.windows import hann, tukey
#
#     # lc = Lightcurve(times, y * hann(len(y)), err=np.ones(len(y)))
    lc = Lightcurve(times, y * tukey(len(y), alpha=0.05), err=np.ones(len(y)))
#     lc = Lightcurve(times, y, err=np.ones(len(y)))
#
#
    ps = Powerspectrum(lc=lc, norm="leahy")
    freqs = ps.freq
    powers = ps.power
# freqs, powers = periodogram(y, fs=sampling_frequency, window="boxcar")

if band_maximum is None:
    band_maximum = freqs[-1] + 1

if band_minimum is None:
    band_minimum = freqs[1]

if plot:
    plt.step(times, y, label="data")
    # plt.plot(times, tukey(len(y), alpha=0.05))

    # plt.plot(times, y, label="flux")
    plt.xlabel("time [s]")
    plt.ylabel("counts")
    plt.show()
    plt.clf()

    plt.loglog()
    plt.step(freqs[1:], powers[1:])
    plt.xlabel("frequency [Hz]")
    plt.ylabel("Power [AU]")
    plt.show()
    plt.clf()

if recovery_mode == "red_noise":
    priors = get_red_noise_prior(sigma_min=sigma_min, sigma_max=sigma_max)
elif recovery_mode == "qpo_plus_red_noise":
    priors = get_red_noise_prior(sigma_min=sigma_min, sigma_max=sigma_max)
    priors.update(get_qpo_prior(frequencies=freqs, min_log_f=np.log(band_minimum), max_log_f=np.log(band_maximum)))#, max_log_width=np.log(0.25), min_log_f=np.log(0.5)))
    priors._resolve_conditions()
elif recovery_mode == "broken_power_law":
    priors = get_broken_power_law_prior(frequencies=freqs)
elif recovery_mode == "pure_qpo":
    priors = get_red_noise_prior(sigma_min=sigma_min, sigma_max=sigma_max)
    priors.update(get_qpo_prior(frequencies=freqs, min_log_f=np.log(band_minimum), max_log_f=np.log(band_maximum)))#, max_log_width=np.log(0.25), min_log_f=np.log(0.5)))
    priors._resolve_conditions()
    del priors["alpha"]
    del priors["log_beta"]
elif recovery_mode == "white_noise":
    priors = get_red_noise_prior(sigma_min=sigma_min, sigma_max=sigma_max)
    del priors["alpha"]
    del priors["log_beta"]
else:
    raise ValueError

if frequency_mask_minimum is None:
    frequency_mask_minimum = 1e-12
if frequency_mask_maximum is None:
    frequency_mask_maximum = freqs[-1] + 1

frequency_mask = [False] * len(freqs)
frequency_mask = np.array(frequency_mask)
idxs = QPOEstimation.utils.get_indices_by_time(minimum_time=frequency_mask_minimum,
                                               maximum_time=frequency_mask_maximum, times=freqs)
frequency_mask[idxs] = True

if recovery_mode == "qpo_plus_red_noise":
    noise_model = "red_noise"
else:
    noise_model = recovery_mode

likelihood = QPOEstimation.likelihood.WhittleLikelihood(
    frequencies=freqs, periodogram=powers, frequency_mask=frequency_mask, noise_model=recovery_mode)
meta_data = dict(kernel_type=recovery_mode, times=times,
                 y=y, likelihood_model=likelihood_model, truths=truths)


result = None
if try_load:
    try:
        result = bilby.result.read_in_result(outdir=f"{outdir}/results", label=label)
    except IOError:
        bilby.utils.logger.info("No result file found. Starting from scratch")
if result is None:
    Path(f"{outdir}/results").mkdir(parents=True, exist_ok=True)
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, outdir=f"{outdir}/results",
                               label=label, sampler="dynesty", nlive=nlive, sample=sample,
                               resume=resume, use_ratio=use_ratio)
result.plot_corner()

max_like_params = result.posterior.iloc[-1]
plt.loglog()
plt.step(freqs[1:], powers[1:], where="mid")
if noise_model == "broken_power_law":
    names_bpl = ["alpha_1", "alpha_2", "log_beta", "log_delta", "rho"]
    max_l_bpl_params = dict()
    for name in names_bpl:
        max_l_bpl_params[name] = max_like_params[name]
    max_l_bpl_params["beta"] = np.exp(max_l_bpl_params["log_beta"])
    max_l_bpl_params["delta"] = np.exp(max_l_bpl_params["log_delta"])
    del max_l_bpl_params["log_delta"]
    del max_l_bpl_params["log_beta"]

if recovery_mode == "qpo_plus_red_noise":
    max_l_psd_no_qpo = QPOEstimation.model.psd.red_noise(freqs[1:], alpha=max_like_params["alpha"],
                                                         beta=np.exp(max_like_params["log_beta"])) + np.exp(
        max_like_params["log_sigma"])
    max_l_psd = max_l_psd_no_qpo + \
                QPOEstimation.model.psd.lorentzian(freqs[1:], amplitude=np.exp(max_like_params["log_amplitude"]),
                                                   central_frequency=np.exp(max_like_params["log_frequency"]),
                                                   width=np.exp(max_like_params["log_width"]))
elif noise_model == "red_noise":
    max_l_psd = QPOEstimation.model.psd.red_noise(freqs[1:], alpha=max_like_params["alpha"],
                                                  beta=np.exp(max_like_params["log_beta"])) + np.exp(
        max_like_params["log_sigma"])
    max_l_psd_no_qpo = max_l_psd
elif noise_model == "broken_power_law":
    max_l_psd = QPOEstimation.model.psd.broken_power_law_noise(freqs[1:], **max_l_bpl_params) + np.exp(
        max_like_params["log_sigma"])
    max_l_psd_no_qpo = max_l_psd
elif noise_model == "pure_qpo":
    max_l_psd_no_qpo = np.exp(max_like_params["log_sigma"]) * np.ones(len(freqs[1:]))
    max_l_psd = max_l_psd_no_qpo + QPOEstimation.model.psd.lorentzian(
        freqs[1:],  amplitude=np.exp(max_like_params["log_amplitude"]),
        central_frequency=np.exp(max_like_params["log_frequency"]), width=np.exp(max_like_params["log_width"]))
elif noise_model == "white_noise":
    max_l_psd = np.exp(max_like_params["log_sigma"]) * np.ones(len(freqs[1:]))
    max_l_psd_no_qpo = max_l_psd
else:
    raise ValueError
plt.loglog(freqs[1:], max_l_psd, label="Max. like. fit")
threshold = -max_l_psd_no_qpo * np.log((1 - 0.9973) / len(freqs[1:]))  # Bonferroni correction
plt.loglog(freqs[1:], threshold, label="3 sigma thres.")
threshold = -max_l_psd_no_qpo * np.log((1 - 0.954) / len(freqs[1:]))  # Bonferroni correction
plt.loglog(freqs[1:], threshold, label="2 sigma thres.")

if recovery_mode == "qpo_plus_red_noise":
    max_like_lorentz = QPOEstimation.model.psd.lorentzian(
        freqs[1:], amplitude=np.exp(max_like_params["log_amplitude"]),
        central_frequency=np.exp(max_like_params["log_frequency"]),
        width=np.exp(max_like_params["log_width"]))
    plt.loglog(freqs[1:], max_like_lorentz, label="Max. like. QPO")
plt.xlabel("frequency [Hz]")
plt.ylabel("Power [arb. units]")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/{label}_max_like_fit.pdf")
plt.show()


if recovery_mode in ["qpo_plus_red_noise", "pure_qpo"]:
    frequency_samples = np.exp(result.posterior["log_frequency"])
    plt.hist(frequency_samples, bins="fd", density=True)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("normalised PDF")
    median = np.median(frequency_samples)
    percentiles = np.percentile(frequency_samples, [16, 84])
    plt.title(
        f"{np.mean(frequency_samples):.2f} + {percentiles[1] - median:.2f} / - {- percentiles[0] + median:.2f}")
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(f"{outdir}/{label}_frequency_posterior.pdf")
    plt.clf()


# clean up
for extension in ["_checkpoint_run.png", "_checkpoint_stats.png", "_checkpoint_trace.png", "_checkpoint_trace_unit.png",
                  "_dynesty.pickle", "_resume.pickle", "_result.json.old", "_samples.dat"]:
    try:
        os.remove(f"{outdir}/results/{label}{extension}")
    except Exception:
        pass
