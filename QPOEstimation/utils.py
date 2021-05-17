likelihood_models = ["gaussian_process", "gaussian_process_windowed"]
modes = ["qpo", "white_noise", "red_noise", "pure_qpo", "general_qpo"]
data_sources = ['injection', 'giant_flare', 'solar_flare', 'grb', 'magnetar_flare', 'hares_and_hounds']
run_modes = ['select_time', 'sliding_window', 'candidates', 'entire_segment', 'from_maximum']
background_models = ["polynomial", "exponential", "fred", "gaussian", "log_normal", "lorentzian", "mean", "skew_gaussian", "fred_norris"]
data_modes = ['normal', 'smoothed', 'smoothed_residual']


class MetaDataAccessor(object):

    """
    Generic descriptor class that allows handy access of properties without long
    boilerplate code. Allows easy access to meta_data dict entries
    """

    def __init__(self, property_name, default=None):
        self.property_name = property_name
        self.container_instance_name = 'meta_data'
        self.default = default

    def __get__(self, instance, owner):
        try:
            return getattr(instance, self.container_instance_name)[self.property_name]
        except KeyError:
            return self.default

    def __set__(self, instance, value):
        getattr(instance, self.container_instance_name)[self.property_name] = value


def get_injection_outdir(injection_mode, recovery_mode, likelihood_model):
    return f"injection/{injection_mode}_injection/{recovery_mode}_recovery/{likelihood_model}"


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
