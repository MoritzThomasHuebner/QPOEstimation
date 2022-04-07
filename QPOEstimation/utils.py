import numpy as np
from typing import Any


class MetaDataAccessor(object):
    """
    Generic descriptor class that allows handy access of properties without long
    boilerplate code. Allows easy access to meta_data dict entries.
    """

    def __init__(self, property_name: str, default: Any = None) -> None:
        self.property_name = property_name
        self.container_instance_name = "meta_data"
        self.default = default

    def __get__(self, instance, owner):
        try:
            return getattr(instance, self.container_instance_name)[self.property_name]
        except KeyError:
            return self.default

    def __set__(self, instance, value):
        getattr(instance, self.container_instance_name)[self.property_name] = value


def get_injection_outdir(
        injection_mode: str, recovery_mode: str, likelihood_model: str,
        base_injection_outdir: str = "injections/injection") -> str:
    """

    Parameters
    ----------
    injection_mode: GP model used to create data. Must be from `QPOEstimation.MODES`.
    recovery_mode: GP model used in the inference process. Must be from `QPOEstimation.MODES`.
    likelihood_model: Likelihood model used in the inference process. Must be from `QPOEstimation.LIKELIHOOD_MODELS`.
    base_injection_outdir: Base string to create the out directory for the injection files.
                           (Default_value = 'injections/injection')

    Returns
    -------
    The directory path.
    """
    return f"{base_injection_outdir}/{injection_mode}_injection/{recovery_mode}_recovery/{likelihood_model}"


def get_injection_label(run_mode: str, injection_id: int, start_time: int = None, end_time: int = None) -> str:
    """ Creates a label for the injection data. This way we consistently name our data and result files.

    Parameters
    ----------
    run_mode: Must be 'entire_segment' or 'select_time'.
    injection_id: Number associated with this simulated data set.
    start_time: If we use 'select_time', the associated start time.
    end_time: If we use 'select_time', the associated end time.

    Returns
    -------
    The label.
    """
    label = f"{str(injection_id).zfill(2)}"
    if run_mode == "entire_segment":
        label += f"_entire_segment"
    elif run_mode == "select_time":
        label += f"_{start_time}_{end_time}"
    return label


def boolean_string(s: str) -> bool:
    """ Convert string True/False argument into a boolean. This is relevant when using argparse for boolean types.

    Parameters
    ----------
    s: A string which is either 'True' or 'False'

    Returns
    -------
    True/False depending on the input string.
    """
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def get_indices_by_time(times: np.ndarray, minimum_time: float, maximum_time: float) -> np.ndarray:
    """ Get time array indices between a minimum and maximum time.

    Parameters
    ----------
    times: The time array.
    minimum_time: The minimum time.
    maximum_time: The maximum time.

    Returns
    -------
    An array with the indices.
    """
    return np.where(np.logical_and(times > minimum_time, times < maximum_time))[0]
