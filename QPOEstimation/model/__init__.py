from inspect import getmembers, isfunction

from . import celerite, psd, series

_functions_list = [o for o in getmembers(series) if isfunction(o[1])]
mean_model_dict = {f[0]: f[1] for f in _functions_list}
