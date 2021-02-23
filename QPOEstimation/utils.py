
class MetaDataAccessor(object):

    """
    Generic descriptor class that allows handy access of properties without long
    boilerplate code. Allows easy access to meta_data dict entries
    """

    def __init__(self, property_name):
        self.property_name = property_name
        self.container_instance_name = 'meta_data'

    def __get__(self, instance, owner):
        try:
            return getattr(instance, self.container_instance_name)[self.property_name]
        except KeyError:
            return None

    def __set__(self, instance, value):
        getattr(instance, self.container_instance_name)[self.property_name] = value


def get_injection_outdir(band, injection_mode, recovery_mode, likelihood_model):
    return f"injection/{band}/{injection_mode}_injection/{recovery_mode}_recovery/{likelihood_model}"
