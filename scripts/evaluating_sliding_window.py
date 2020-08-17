import bilby

for i in range(31):
    try:
        res = bilby.result.read_in_result(f"sliding_window/{i}_result.json")
        print(f"{i}: {res.log_bayes_factor}")
    except Exception:
        pass