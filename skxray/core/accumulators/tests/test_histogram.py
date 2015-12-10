import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from skxray.core.accumulators.histogram import Histogram
from time import time


def _1d_histogram_tester(binlowhighs, x, weights=1):
    h = Histogram(binlowhighs)
    h.fill(x, weights=weights)
    if np.isscalar(weights):
        ynp = np.histogram(x, h.edges[0])[0]
    else:
        ynp = np.histogram(x, h.edges[0], weights=weights)[0]

    assert_array_almost_equal(ynp, h.values)


def test_1d_histogram():
    binlowhigh = [10, 0, 10.01]
    xf = np.random.random(1000000)*40
    xi = xf.astype(int)
    wf = np.linspace(1, 10, len(xf))
    wi = wf.copy()
    vals = [
        [binlowhigh, xf, wf],
        [binlowhigh, xf, 1],
        [binlowhigh, xi, wi],
        [binlowhigh, xi, 1],
        [binlowhigh, xf, wi],
        [binlowhigh, xi, wf],
    ]
    for binlowhigh, x, w in vals:
        yield _1d_histogram_tester, binlowhigh, x, w



def _2d_histogram_tester(binlowhighs, x, y, weights=1):
    h = Histogram(*binlowhighs)
    h.fill(x, y, weights=weights)
    if np.isscalar(weights):
        ynp = np.histogram2d(x, y, bins=h.edges)[0]
    else:
        ynp = np.histogram2d(x, y, bins=h.edges, weights=weights)[0]

    assert_array_almost_equal(ynp, h.values)


def test_2d_histogram():
    ten = [10, 0, 10.01]
    nine = [9, 0, 9.01]
    xf = np.random.random(1000000)*40
    yf = np.random.random(1000000)*40
    xi = xf.astype(int)
    yi = yf.astype(int)
    wf = np.linspace(1, 10, len(xf))
    wi = wf.copy()
    vals = [
        [[ten, ten], xf, yf, wf],
        [[ten, nine], xf, yf, 1],
        [[ten, ten], xi, yi, wi],
        [[ten, ten], xi, yi, 1],
        [[ten, nine], xf, yf, wi],
        [[ten, nine], xi, yi, wf],
    ]
    for binlowhigh, x, y, w in vals:
        yield _2d_histogram_tester, binlowhigh, x, y, w

import itertools
if __name__ == '__main__':
    x = [1000, 0, 10.01]
    y = [1000, 0, 9.01]
    xf = np.random.random(1000000)*10
    yf = np.random.random(1000000)*9
    xi = xf.astype(int)
    yi = yf.astype(int)
    wf = np.linspace(1, 10, len(xf))
    wi = wf.copy()

    print("Testing 2D histogram timings")
    for xvals, yvals, weights in itertools.product([xf, xi], [yf, yi], [wf, wi]):
        print('xvals are of type {}'.format(xvals.dtype))
        print('yvals are of type {}'.format(yvals.dtype))
        print('weights are of type {}'.format(weights.dtype))
        t0 = time()
        h = Histogram(x, y)
        h.fill(xvals, yvals, weights=weights)
        skxray_time = time() - t0

        print('skxray = {}'.format(skxray_time))
        edges = h.edges
        t0 = time()
        ynp = np.histogram2d(xvals, yvals, bins=edges, weights=weights)[0]
        numpy_time = time() - t0
        print('numpy = {}'.format(numpy_time))
        print('skxray is {} faster than numpy'.format(numpy_time / skxray_time))
        assert_almost_equal(np.sum(h.values), np.sum(ynp))
    #
    # test_1d_histogram()
    # test_2d_histogram()

#TODO do a better job sampling the variable space
