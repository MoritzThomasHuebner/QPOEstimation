import numpy as np


def moving_average(ys, n=64):
    res = np.zeros(len(ys))
    for i in range(len(ys)):
        if i < n:
            total_bins = i + n
            res[i] = np.mean(ys[:total_bins])
        elif i > len(ys) - n:
            total_bins = i - n
            res[i] = np.mean(ys[total_bins:])
        else:
            indices = np.arange(i - n, i + n, 1)
            res[i] = np.mean(ys[indices])
    return res


def exponential_smoothing(ys, alpha):
    s = np.zeros(len(ys))
    s[0] = ys[0]
    for i in range(1, len(ys)):
        s[i] = alpha * ys[i] + s[i-1]*(1-alpha)
    return s


def second_order_exponential_smoothing(ys, alpha, beta):
    s = np.zeros(len(ys))
    b = np.zeros(len(ys))
    s[0] = ys[0]
    b[0] = ys[1] - ys[0]
    for i in range(1, len(ys)):
        s[i] = alpha * ys[i] + (1 - alpha)*(s[i-1] - b[i-1])
        b[i] = beta * (s[i] - s[i-1]) + (1 - beta)*b[i-1]
    return s


def two_sided_exponential_smoothing(ys, alpha):
    forward = exponential_smoothing(ys, alpha)
    backwards = exponential_smoothing(ys[::-1], alpha)[::-1]
    res = (forward + backwards)/2
    return res


def two_sided_second_order_exponential_smoothing(ys, alpha, beta):
    forward = second_order_exponential_smoothing(ys, alpha, beta)
    backwards = second_order_exponential_smoothing(ys[::-1], alpha, beta)[::-1]
    res = (forward + backwards)/2
    return res


def boxcar_filter(ys, n):
    res = np.zeros(len(ys))
    for i, c in enumerate(ys):
        if i < (n - 1)/2:
            boxcar = ys[0: int(i + (n - 1) / 2)]
        elif i > len(ys) - (n - 1)/2:
            boxcar = ys[int(i - (n - 1) / 2):-1]
        else:
            boxcar = ys[int(i - (n - 1) / 2): int(i + (n - 1) / 2)]
        res[i] = sum(boxcar) / len(boxcar)

    return res
