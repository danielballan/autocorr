# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################
"""
This is the module for calibration functions and data
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import scipy.signal
from collections import deque
from nsls2.constants import calibration_standards
from nsls2.feature import (filter_peak_height, peak_refinement,
                           refine_log_quadratic)
from nsls2.core import (warp_to_phi, warp_to_radius,
                        pairwise, bin_edges_to_centers, bin_1D)

def estimate_d_blind(name, wavelength, bin_centers, ring_average,
               window_size, threshold, max_peak_count=None):
    """ Estimate the sample-detector distance

    Given a radially integrated calibration image return an estimate for
    the sample-detector distance.  This function does not require a
    rough estimate of what d should be.

    For the peaks found the detector-sample distance is estimated via
    .. math ::

        d = \\frac{m}{\\tan 2\\theta}

    where :math:`m` is the distance in mm from the calibrated center
    to the ring on the detector.

    Parameters
    ----------
    name : str
        The name of the calibration standard.  Used to look up the
        expected peak location

    wavelength : float
        The wavelength of scattered x-ray in nm

    bin_centers : array
        The distance from the calibrated center to the center of
        the ring's annulus in mm

    ring_average : array
        The average intensity in the given ring.  In counts [arb]

    window_size : int
        The number of elements on either side of a local maximum to
        use for locating and refining peaks.  Candidates are identified
        as a relative maximum in a window sized (2*window_size + 1) and
        the same window is used for fitting the peaks to refine the location.

    threshold : float
        The minimum range a peak needs to have in the window to be accepted
        as a real peak.  This is used to filter out the (many) spurious local
        maximum which can be found.

    max_peak_count : int, optional
        Use at most this many peaks

    Returns
    -------
    dist_sample : float
        The detector-sample distance in mm.  This is the mean of the estimate
        from all of the peaks used.

    std_dist_sample : float
        The standard deviation of the d estimated by each of the peaks

    """
    if max_peak_count is None:
        max_peak_count = np.iinfo(int).max
    # TODO come up with way to estimate threshold blind, maybe otsu

    # get the calibration standard
    cal = calibration_standards[name]
    # find the local maximums
    cands = scipy.signal.argrelmax(ring_average, order=window_size)[0]
    # filter local maximums by size
    cands = filter_peak_height(ring_average, cands,
                               threshold, window=window_size)
    # TODO insert peak identification validation.  This might be better than
    # improving the threshold value.

    # refine the locations of the peaks
    peaks_x, peaks_y = peak_refinement(bin_centers, ring_average, cands,
                                       window_size, refine_log_quadratic)
    # compute tan(2theta) for the expected peaks
    tan2theta = np.tan(cal.convert_2theta(wavelength))
    # figure out how many peaks we can look at
    slc = slice(0, np.min([len(tan2theta), len(peaks_x), max_peak_count]))
    # estimate the sample-
    d_array = (peaks_x[slc] / tan2theta[slc])

    return np.mean(d_array), np.std(d_array)


def refine_center(image, calibrated_center, pixel_size, phi_steps,
                  nx=750, min_x=10, max_x=60, window_size=5,
                  threshold=17000, max_peaks=7):
    """Refines the location of the center of the beam.

    This relies on being able to see the whole powder pattern.

    Parameters
    ----------
    image : ndarray
        The image
    calibrated_center : tuple
        (row, column) the estimated center
    pixel_size : tuple
        (pixel_height, pixel_width)
    phi_steps : int
        How many regions to split the ring into, should be >10
    nx : int
        Number of bins to use for radial binning
    min_x : float
        The minimum radius to use for radial binning
    max_x : float
        The maximum radius to use for radial binning
    window_size : int
        The window size to use (in bins) to use when refining peaks
    threshold : float
        Minimum peaks size
    max_peaks : int
        Number of rings to look it

    Returns
    -------
    calibrated_center : tuple
        The refined calibrated center.
    """
    phi = warp_to_phi(image.shape, calibrated_center, pixel_size).ravel()
    r = warp_to_radius(image.shape, calibrated_center, pixel_size).ravel()
    I = image.ravel()

    phi_steps = np.linspace(-np.pi, np.pi, phi_steps, endpoint=True)
    out = deque()
    for phi_start, phi_end in pairwise(phi_steps):
        mask = (phi <= phi_end) * (phi > phi_start)
        out.append(bin_1D(r[mask], I[mask],
                          nx=nx, min_x=min_x, max_x=max_x))
    out = list(out)

    ring_trace = []
    for bins, b_sum, b_count in out:
        mask = b_sum > 0
        avg = b_sum[mask] / b_count[mask]
        bin_centers = bin_edges_to_centers(bins)[mask]

        cands = scipy.signal.argrelmax(avg, order=window_size)[0]
        # filter local maximums by size
        cands = filter_peak_height(avg, cands,
                                   threshold, window=window_size)
        ring_trace.append(bin_centers[cands[:max_peaks]])

    print([len(_) for _ in ring_trace])
    ring_trace = np.vstack(ring_trace).T

    mean_dr = np.mean(ring_trace - np.mean(ring_trace, axis=1, keepdims=True), axis=0)

    phi_centers = bin_edges_to_centers(phi_steps)

    delta = np.mean(np.diff(phi_centers))
    # this is doing just one term of a Fourier series
    # note that we have to convert _back_ to pixels from real units
    # TODO do this with better integration/handle repeat better
    col_shift = np.sum(np.sin(phi_centers) * mean_dr) * delta / (np.pi * pixel_size[1])
    row_shift = np.sum(np.cos(phi_centers) * mean_dr) * delta / (np.pi * pixel_size[0])

    return tuple(np.array(calibrated_center) + np.array([row_shift, col_shift]))
