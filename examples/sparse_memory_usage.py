import numpy as np
import sparse
import autocorr
import autocorr.multitau_sparse
import os
import psutil


process = psutil.Process(os.getpid())

def sizeof_fmt(num, suffix='B'):
    # https://stackoverflow.com/a/1094933/1221924
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def usage():
    return sizeof_fmt(process.memory_info().rss)

print(f"Memory usage: {usage()} (baseline)")

N = 2048 ** 2
np.random.seed(0)
t = np.arange(N)
a = np.exp(-0.05 * t)[:, np.newaxis] + np.random.rand(N, 24) * 0.1
DENSE_FRACTION = 0.01
zero_mask = np.random.rand(*a.shape) > DENSE_FRACTION
a[zero_mask] = 0
result = autocorr.multitau(a)
print(f"Memory usage: {usage()} (dense)")
del result
s = sparse.COO.from_numpy(a)
del a
autocorr.multitau_sparse.multitau(s)
print(f"Memory usage: {usage()} (sparse)")
