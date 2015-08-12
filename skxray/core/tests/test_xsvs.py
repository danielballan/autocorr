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
from __future__ import absolute_import, division, print_function
import logging

import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal)

from skimage.morphology import convex_hull_image

import skxray.core.correlation as corr
import skxray.core.xsvs as xsvs
import skxray.core.roi as roi
from skxray.testing.decorators import skip_if

logger = logging.getLogger(__name__)


def test_xsvs():
    images = []
    for i in range(10):
        int_array = np.tril(i*np.ones(10))
        if i==10/2:
            int_array[int_array == 0] = 20
        else:
            int_array[int_array == 0] = i*2
        images.append(int_array)

    images = np.asarray(images)
    roi_data = np.array(([4, 2, 2, 2], [0, 5, 2, 2]), dtype=np.int64)
    label_array = roi.rectangles(roi_data, shape=images[0].shape)

    num_times = 4
    num_rois = 2


def test_normalize_bin_edges():
    num_times = 3
    num_rois = 2
    mean_roi = np.arear([2.5, 4.0])
    max_cts = 5

    bin_edges, bin_cen = xsvs.normalize_bin_edges(num_times, num_rois,
                                                  mean_roi, max_cts)

    assert_array_almost_equal(bin_edges[0, 0], np.array([0., 0.4, 0.8,
                                                        1.2, 1.6]))

    assert_array_almost_equal(bin_edges[2, 1], np.array([0., 0.0625, 0.125,
                                                         0.1875, 0.25, 0.3125,
                                                         0.375 , 0.4375, 0.5,
                                                         0.5625, 0.625, 0.6875,
                                                         0.75, 0.8125, 0.875,
                                                         0.9375, 1., 1.0625,
                                                         1.125, 1.1875]))

    assert_array_almost_equal(bin_cen[0, 0], np.array([0.2, 0.6, 1., 1.4]))
