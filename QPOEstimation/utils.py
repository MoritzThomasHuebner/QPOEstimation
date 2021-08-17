import numpy as np

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


def get_injection_outdir(injection_mode, recovery_mode, likelihood_model, base_injection_outdir='injection'):
    return f"{base_injection_outdir}/{injection_mode}_injection/{recovery_mode}_recovery/{likelihood_model}"


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_indices_by_time(times, minimum_time, maximum_time):
    return np.where(np.logical_and(times > minimum_time, times < maximum_time))[0]
