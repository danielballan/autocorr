# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
"""
This module is for spectroscopy specific tools (spectrum fitting etc).
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import zip
import numpy as np
from scipy.integrate import simps


def fit_quad_to_peak(x, y):
    """
    Fits a quadratic to the data points handed in
    to the from y = b[0](x-b[1])**2 + b[2] and R2
    (measure of goodness of fit)

    Parameters
    ----------
    x : ndarray
        locations
    y : ndarray
        values

    Returns
    -------
    b : tuple
       coefficients of form y = b[0](x-b[1])**2 + b[2]

    R2 : float
      R2 value

    """

    lenx = len(x)

    # some sanity checks
    if lenx < 3:
        raise Exception('insufficient points handed in ')
    # set up fitting array
    X = np.vstack((x ** 2, x, np.ones(lenx))).T
    # use linear least squares fitting
    beta, _, _, _ = np.linalg.lstsq(X, y)

    SSerr = np.sum(np.power(np.polyval(beta, x) - y, 2))
    SStot = np.sum(np.power(y - np.mean(y), 2))
    # re-map the returned value to match the form we want
    ret_beta = (beta[0],
                -beta[1] / (2 * beta[0]),
                beta[2] - beta[0] * (beta[1] / (2 * beta[0])) ** 2)

    return ret_beta, 1 - SSerr / SStot


def align_and_scale(energy_list, counts_list, pk_find_fun=None):
    """

    Parameters
    ----------
    energy_list : iterable of ndarrays
        list of ndarrays with the energy of each element

    counts_list : iterable of ndarrays
        list of ndarrays of counts/element

    pk_find_fun : function or None
       A function which takes two ndarrays and returns parameters
       about the largest peak.  If None, defaults to `find_largest_peak`.
       For this demo, the output is (center, height, width), but this sould
       be pinned down better.

    Returns
    -------
    out_e : list of ndarray
       The aligned/scaled energy arrays

    out_c : list of ndarray
       The count arrays (should be the same as the input)
    """
    if pk_find_fun is None:
        pk_find_fun = find_larest_peak

    base_sigma = None
    out_e, out_c = [], []
    for e, c in zip(energy_list, counts_list):
        E0, max_val, sigma = pk_find_fun(e, c)
        print(E0, max_val, sigma)
        if base_sigma is None:
            base_sigma = sigma
        out_e.append((e - E0) * base_sigma / sigma)
        out_c.append(c)

    return out_e, out_c


def find_larest_peak(X, Y, window=5):
    """
    Finds and estimates the location, width, and height of
    the largest peak. Assumes the top of the peak can be
    approximated as a Gaussian.  Finds the peak properties
    using least-squares fitting of a parabola to the log of
    the counts.

    The region around the peak can be approximated by
    Y = Y0 * exp(- (X - X0)**2 / (2 * sigma **2))

    Parameters
    ----------
    X : ndarray
       The independent variable

    Y : ndarary
      Dependent variable sampled at positions X

    window : int, optional
       The size of the window around the maximum to use
       for the fitting


    Returns
    -------
    X0 : float
        The location of the peak

    Y0 : float
        The magnitude of the peak

    sigma : float
        Width of the peak
    """

    # make sure they are _really_ arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # get the bin with the largest number of counts
    j = np.argmax(Y)
    roi = slice(np.max(j - window, 0),
                j + window + 1)

    (w, X0, Y0), R2 = fit_quad_to_peak(X[roi],
                                        np.log(Y[roi]))

    return X0, np.exp(Y0), 1/np.sqrt(-2*w)


def integrate_ROI(x_value_array, counts, x_min, x_max):
    """
    Integrate region(s) of the given spectrum.  If `x_min`
    and `x_max` are arrays/lists they must be equal in length.
    The values contained in the 'x_value_array' must have monotonic (equal)
    spacing and be increasing from left to right.
    The weight from each of the regions is summed.

    This returns a single scalar value for the integration.

    Currently this code integrates from the left edge
    of the first bin fully contained in the range to
    the right edge of the last bin partially contained
    in the range.  This may produce bias and should be
    addressed when this is an issue.

    Parameters
    ----------
    counts : array
        Counts in spectrum, any units

    x_value_array : array
        The array of all x values corresponding to the left (lower) 
        edge of each bin in the spectrum. Values must be monotonically spaced.

    x_min : float or array
        The lower edge of the integration region

    x_max : float or array
        The upper edge of the integration region

    Returns
    -------
    float
        The integrated intensity in same units as `counts`
    """
    # make sure x_value_array (x-values) and counts (y-values) are arrays
    x_value_array = np.asarray(x_value_array)
    counts = np.asarray(counts)
    
    # make sure values in x_value_array increase in value
    if not np.all(np.diff(x_value_array) > 0):
        x_value_array = x_value_array[::-1]
    
    # make sure values in x_value_array are equally spaced
    if not np.all((np.diff(x_value_array) / (x_value_array[1] - 
                                             x_value_array[0])) == 1):
        raise ValueError("Energy must be monotonically increasing")

    # up-cast to 1d and make sure it is flat
    x_min = np.atleast_1d(x_min).ravel()
    x_max = np.atleast_1d(x_max).ravel()

    # sanity checks on integration bounds
    if len(x_min) != len(x_max):
        raise ValueError("integration bounds must have same lengths")

    if np.any(x_min >= x_max):
        raise ValueError("lower integration bound must be less than "
                         "upper integration bound ")

    if np.any(x_min <= x_value_array[0]):
        raise ValueError("lower integration boundary values must be greater"
                         "than, or equal to the lowest value in spectrum range")

    if np.any(x_max >= x_value_array[-1]):
        raise ValueError("upper integration boundary values must be less "
                         "than, or equal to the highest value in the spectrum "
                         "range")

    # find the bottom index of each integration bound
    bottom_indx = x_value_array.searchsorted(x_min)
    # find the top index of each integration bound
    # NOTE: Why does this expression include '+1'??
    # Aren't the x_min and x_max values supplied in pairs? Why is there an additional offset?
    top_indx = x_value_array.searchsorted(x_max) + 1

    # set up temporary variables
    accum = 0
    # integrate each region
    for bot, top in zip(bottom_indx, top_indx):
        accum += simps(counts[bot:top], x_value_array[bot:top])

    return accum
