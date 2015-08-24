# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Developed at the NSLS-II, Brookhaven National Laboratory             #
# Developed by Sameera K. Abeykoon and Yugang Zhang, June 2015         #
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
    X-ray speckle visibility spectroscopy(XSVS) - Dynamic information of
    the speckle patterns  are obtained by analyzing the photon statistics
    and calculating the speckle contrast in single scattering patterns.

    This module will provide XSVS analysis tools
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import time

import skxray.core.correlation as corr
import skxray.core.roi as roi
from skxray.core.utils import bin_edges_to_centers, geometric_series

import logging
logger = logging.getLogger(__name__)


def xsvs(image_sets, label_array, timebin_num=2, number_of_img=50,
         max_cts=None):
    """
    This function will provide the probability density of detecting photons
    for different integration time.

    The experimental probability density P(K) of detecting photons K is
    obtained by histogramming the photon counts over an ensemble of
    equivalent pixels and over a number of speckle patterns recorded
    with the same integration time T under the same condition.

    Parameters
    ----------
    image_sets : array
        sets of images
    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).
    timebin_num : int, optional
        integration times
    number_of_img : int, optional
        number of images
    max_cts : int, optional
       the brightest pixel in any ROI in any image in the image set.

    Returns
    -------
    prob_k_all : array
        probability density of detecting photons
    prob_k_std_dev : array
        standard deviation of probability density of detecting photons

    Note
    ----
    These implementation is based on following references
    References: text [1]_, text [2]_

    .. [1] L. Li, P. Kwasniewski, D. Oris, L Wiegart, L. Cristofolini,
       C. Carona and A. Fluerasu , "Photon statistics and speckle visibility
       spectroscopy with partially coherent x-rays" J. Synchrotron Rad.,
       vol 21, p 1288-1295, 2014.

    .. [2] R. Bandyopadhyay, A. S. Gittings, S. S. Suh, P.K. Dixon and
       D.J. Durian "Speckle-visibilty Spectroscopy: A tool to study
       time-varying dynamics" Rev. Sci. Instrum. vol 76, p  093110, 2005.

    """
    if max_cts is None:
        max_cts = roi.roi_max_counts(image_sets, label_array)

    # number of ROI's
    num_roi = np.max(label_array)

    # create integration times
    time_bin = geometric_series(timebin_num, number_of_img)

    # number of items in the time bin
    num_times = len(time_bin)

    # find the label's and pixel indices for ROI's
    labels, indices = corr.extract_label_indices(label_array)

    # number of pixels per ROI
    num_pixels = np.bincount(labels, minlength=(num_roi+1))[1:]

    # probability density of detecting speckles
    prob_k_all = np.zeros([num_times, num_roi], dtype=np.object)
    # square of probability density of detecting speckles
    prob_k_pow_all = np.zeros([num_times, num_roi], dtype=np.object)
    # standard deviation of probability density of detecting photons
    prob_k_std_dev = np.zeros([num_times, num_roi], dtype=np.object)

    # get the bin edges for each time bin for each ROI
    bin_edges = np.zeros((num_times, num_roi), dtype=object)
    for i in range(num_times):
        for j in range(num_roi):
            bin_edges[i, j] = np.arange(max_cts*2**i)

    start_time = time.time()  # used to log the computation time (optionally)

    for i, images in enumerate(image_sets):
        # Ring buffer, a buffer with periodic boundary conditions.
        # Images must be keep for up to maximum delay in buf.
        buf = np.zeros([num_times, timebin_num],
                       dtype=np.object)  # matrix of buffers

        # to track processing each time level
        track_level = np.zeros(num_times)

        # to increment buffer
        cur = np.ones(num_times)*timebin_num

        # to track how many images processed in each level
        img_per_level = np.zeros(num_times, dtype=np.int64)

        prob_k = np.zeros([num_times, num_roi], dtype=np.object)
        prob_k_pow = np.zeros([num_times, num_roi], dtype=np.object)

        for n, img in enumerate(images):
            cur[0] = (1 + cur[0]) % timebin_num
            # read each frame
            # Put the image into the ring buffer.
            buf[0, cur[0] - 1] = (np.ravel(img))[indices]

            _process(num_roi, 0, cur[0] - 1, buf, img_per_level, labels,
                     max_cts, bin_edges[0, 0], prob_k, prob_k_pow)

            # check whether the number of levels is one, otherwise
            # continue processing the next level
            processing = num_times > 1
            level = 1

            while processing:
                if not track_level[level]:
                    track_level[level] = 1
                    processing = 0
                else:
                    prev = 1 + (cur[level - 1] - 2) % timebin_num
                    cur[level] = 1 + cur[level] % timebin_num

                    buf[level, cur[level]-1] = (buf[level-1,
                                                    prev-1] +
                                                buf[level-1,
                                                    cur[level - 1] - 1])
                    track_level[level] = 0

                    _process(num_roi, level, cur[level]-1, buf, img_per_level,
                             labels, max_cts, bin_edges[level, 0], prob_k,
                             prob_k_pow)
                    level += 1
                    # Checking whether there is next level for processing
                    processing = level < num_times

            prob_k_all += (prob_k - prob_k_all)/(i + 1)
            prob_k_pow_all += (prob_k_pow - prob_k_pow_all)/(i + 1)

            prob_k_std_dev = np.power((prob_k_pow_all -
                                       np.power(prob_k_all, 2)), .5)

    # ending time for the process
    end_time = time.time()

    logger.info("Processing time for XSVS took {0} seconds."
                "".format(end_time - start_time))
    return prob_k_all, prob_k_std_dev


def _process(num_roi, level, buf_no, buf, img_per_level, labels, max_cts,
             bin_edges, prob_k, prob_k_pow):
    """
    Internal helper function. This modifies inputs in place.

    This helper function calculate probability of detecting speckles for
    each integration time.

    Parameters
    ----------
    num_roi : int
        number of ROI's
    level : int
        current time level(integration time)
    buf_no : int
        current buffer number
    buf : array
        image data array to use for XSVS
    img_per_level : int
        to track how many images processed in each level
    labels : array
        labels of the required region of interests(ROI's
    max_cts: int
        maximum pixel count
    bin_edges : array
        bin edges for each integration times and each ROI
    prob_k : array
        probability density of detecting speckles
    prob_k_pow : array
        squares of probability density of detecting speckles
    """
    img_per_level[level] += 1

    for j in range(num_roi):
        roi_data = buf[level, buf_no][labels == j+1]

        spe_hist, bin_edges = np.histogram(roi_data, bins=bin_edges,
                                           normed=True)

        prob_k[level, j] += (np.nan_to_num(spe_hist) -
                             prob_k[level, j])/(img_per_level[level])

        prob_k_pow[level, j] += (np.power(np.nan_to_num(spe_hist), 2) -
                                 prob_k_pow[level, j])/(img_per_level[level])

    return  # modifies arguments in place!


def normalize_bin_edges(num_times, num_rois, mean_roi, max_cts):
    """
    This will provide the normalize bin edges and bin centers for each
    integration times.

    Parameters
    ----------
    num_times : int
        number of integration times for XSVS
    num_rois : int
        number of ROI's
    mean_roi : array
        mean intensity of each ROI
        shape (number of ROI's)
    max_cts : int
        maximum pixel counts

    Returns
    -------
    norm_bin_edges : array
        normalized speckle count bin edges
         shape (num_times, num_rois)
    norm_bin_centers :array
        normalized speckle count bin centers
        shape (num_times, num_rois)
    """
    norm_bin_edges = np.zeros((num_times, num_rois), dtype=object)
    norm_bin_centers = np.zeros((num_times, num_rois), dtype=object)
    for i in range(num_times):
        for j in range(num_rois):
            norm_bin_edges[i, j] = np.arange(max_cts*2**i)/(mean_roi[j]*2**i)
            norm_bin_centers[i, j] = bin_edges_to_centers(norm_bin_edges[i, j])

    return norm_bin_edges, norm_bin_centers
