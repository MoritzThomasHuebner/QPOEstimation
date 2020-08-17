import bilby
im
for i in range(31):
    mean_log_bf = 0
    for j in range(9):
        try:
            res = bilby.result.read_in_result(f"sliding_window/period_{j}/{i}_result.json")
            mean_log_bf += res.log_bayes_factor
            if res.log_bayes_factor > 4:
                print(f"{j} {i}: {res.log_bayes_factor}")
        except Exception:
            pass
    # print(f"{i}: {mean_log_bf}")
