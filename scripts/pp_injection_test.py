import bilby

res_list = [bilby.result.read_in_result(f'sliding_window_5_64Hz_one_qpo_injections/one_qpo/results/{str(injection_id).zfill(2)}_result.json') for injection_id in range(100)]

bilby.result.make_pp_plot(results=res_list, outdir='.')
