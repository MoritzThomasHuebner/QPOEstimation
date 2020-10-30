import bilby
import json

res_list = []
for injection_id in range(0, 100):
    with open(f'injection_files/one_qpo/{str(injection_id).zfill(2)}_params.json') as f:
        injection_params = json.load(f)
    res = bilby.result.read_in_result(f'sliding_window_5_64Hz_one_qpo_injections/one_qpo/results/{str(injection_id).zfill(2)}_result.json')
    res.injection_parameters = injection_params
    res_list.append(res)

bilby.result.make_pp_plot(results=res_list, filename='pp_test_one_qpo.png')

res_list = []
for injection_id in range(0, 100):
    with open(f'injection_files/no_qpo/{str(injection_id).zfill(2)}_params.json') as f:
        injection_params = json.load(f)
    print(injection_params)
    res = bilby.result.read_in_result(f'sliding_window_5_64Hz_no_qpo_injections/no_qpo/results/{str(injection_id).zfill(2)}_result.json')
    res.injection_parameters = injection_params
    res_list.append(res)
    print(res)

bilby.result.make_pp_plot(results=res_list, filename='pp_test_no_qpo.png')
