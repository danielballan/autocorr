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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import logging
logger = logging.getLogger(__name__)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)
import sys

from nose.tools import assert_equal, assert_true, raises

import skxray.diff_roi_choice as diff_roi

from skxray.testing.decorators import known_fail_if
import numpy.testing as npt


def test_roi_rectangles():
    num_rois = 3
    detector_size = (15, 26)
    roi_data = np.array(([2, 2, 6, 3], [6, 7, 8, 5], [8, 18, 5, 10]),
                        dtype=np.int64)

    roi_inds, num_pixels, pixel_list = diff_roi.roi_rectangles(num_rois,
                                                               roi_data,
                                                               detector_size)
    ty = np.zeros(detector_size).ravel() * np.nan
    ty[pixel_list] = roi_inds
    ty[np.isnan(ty)] = 0
    num_pixels_m = (np.bincount(ty.astype(int)))[1:]

    re_mesh = ty.reshape(*detector_size)
    for i, (col_coor, row_coor, col_val, row_val) in enumerate(roi_data, 0):
        ind_co = np.column_stack(np.where(re_mesh == i + 1))

        left, right = np.max([col_coor, 0]), np.min([col_coor + col_val,
                                                     detector_size[0]])
        top, bottom = np.max([row_coor, 0]), np.min([row_coor + row_val,
                                                     detector_size[1]])
        assert_almost_equal(left, ind_co[0][0])
        assert_almost_equal(right-1, ind_co[-1][0])
        assert_almost_equal(top, ind_co[0][1])
        assert_almost_equal(bottom-1, ind_co[-1][-1])

    assert_array_equal(num_pixels, num_pixels_m)


"""def test_roi_rings():
    calibrated_center = (6., 4.)
    img_dim = (20, 25)
    first_q = 2
    delta_q = 3
    num_qs = 10  # number of Q rings

    (q_inds, q_ring_val, num_pixels,
     pixel_list) = diff_roi.roi_rings(img_dim, calibrated_center, num_qs,
                                      first_q, delta_q)

    ty = np.ones(img_dim).ravel() * np.nan
    ty[pixel_list] = q_inds
    ty[np.isnan(ty)] = 0

    num_pixels_m = (np.bincount(ty.astype(int)))[1:]
    assert_array_equal(num_pixels, num_pixels_m)

    xx, yy = np.mgrid[:img_dim[0], :img_dim[1]]
    x_ = (xx - calibrated_center[0])
    y_ = (yy - calibrated_center[1])
    grid_values = np.float_(np.hypot(x_, y_))"""

"""q_ring_val_m = np.array([[2.5, 4.5],
                             [4.5, 6.5],
                             [6.5, 8.5],
                             [8.5, 10.5],
                             [10.5, 12.5],
                             [12.5, 14.5],
                             [14.5, 16.5],
                             [16.5, 18.5],
                             [18.5, 20.5],
                             [20.5, 22.5]])

    num_pixels_m = np.array([48, 68, 60, 61, 61, 65, 47, 44, 18, 7])

    i = 1
    for r in range(0, num_qs):
        if q_ring_val[r][0]<=np.any(grid_values)<=q_ring_val[r][1]:
            grid_values == i
        i += 1

    print (grid_values)
    q_inds_m = np.array([3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5,
                         5, 6, 6, 6, 7, 7, 8, 8, 9, 3, 3, 2, 2, 2, 2, 2,
                         2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8,
                         8, 9, 3, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3,
                         4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 3, 2, 2, 1, 1,
                         1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6,
                         7, 7, 8, 8, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3,
                         4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 2, 2, 1, 1, 1, 1,
                         2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 2, 2,
                         1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                         8, 8, 2, 2, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                         6, 6, 7, 7, 8, 8, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2,
                         3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 3, 2, 2, 1,
                         1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6,
                         6, 7, 7, 8, 8, 3, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2,
                         2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 3, 3,
                         2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5,
                         6, 6, 7, 7, 8, 8,  9,  3,  3,  3,  3,  2,  2,
                          2,  2,  2,  3,
        3,  3,  3,  4,  4,  5,  5,  6,  6,  6,  7,  7,  8,  8,  9,  4,  4,
        3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  5,  5,  5,  6,  6,
        7,  7,  8,  8,  8,  9,  4,  4,  4,  4,  3,  3,  3,  3,  3,  4,  4,
        4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  8,  8,  9,  9,  5,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,
        8,  8,  8,  9,  9,  5,  5,  5,  4,  4,  4,  4,  4,  4,  4,  5,  5,
        5,  5,  6,  6,  6,  7,  7,  7,  8,  8,  9,  9, 10,  6,  5,  5,  5,
        5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  8,  8,
        9,  9,  9, 10,  6,  6,  6,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,
        6,  6,  7,  7,  7,  8,  8,  8,  9,  9, 10, 10,  6,  6,  6,  6,  6,
        6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,
       10, 10, 10])



    #assert_array_almost_equal(q_ring_val_m, q_ring_val)

    #assert_array_equal(q_inds, np.ravel(q_inds_m))"""


"""def test_roi_rings_step():
    calibrated_center = (4., 6.)
    img_dim = (20, 25)
    first_q = 2.5
    delta_q = 2

    # using a step for the Q rings
    num_qs = 6  # number of Q rings
    step_q = 1  # step value between each Q ring

    (qstep_inds, qstep_ring_val, numstep_pixels,
     pixelstep_list) = diff_roi.roi_rings_step(img_dim, calibrated_center,
                                               num_qs, first_q, delta_q,
                                               step_q)
    ty = np.ones(img_dim).ravel() * np.nan
    ty[pixelstep_list] = qstep_inds
    ty[np.isnan(ty)] = 0

    numstep_pixels_m = (np.bincount(ty.astype(int)))[1:]
    assert_array_equal(numstep_pixels, numstep_pixels_m)"""

"""qstep_inds_m = np.array([1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                             1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                             6, 6, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                             6, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                             1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1,
                             1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 1,
                             1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 1, 1,
                             1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1,
                             1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 2,
                             2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 2, 2, 2,
                             2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6,
                             6, 6, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                             5, 5, 6, 6, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 3,
                             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5,
                             5, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
                             5, 5, 6, 6, 6, 4, 4, 4, 4, 5, 5, 6, 6, 4, 4,
                             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6,
                             6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6,
                             6, 6, 5, 5, 5, 5, 6, 6, 6, 5, 5, 5, 5, 5, 5,
                             5, 5, 5, 5, 5, 5, 6, 6, 6, 6])

    numstep_pixels_m = np.array([48, 70, 61, 67, 47, 34])

    qstep_ring_val_m = np.array([[2.5, 4.5],
                                 [5.5, 7.5],
                                 [8.5, 10.5],
                                 [11.5, 13.5],
                                 [14.5, 16.5],
                                 [17.5, 19.5]])

    assert_almost_equal(qstep_ring_val, qstep_ring_val_m)

    #assert_array_equal(qstep_inds, np.ravel(qstep_inds_m))"""


"""def test_roi_rings_diff_steps():
    calibrated_center = (10., 4.)
    img_dim = (45, 25)
    first_q = 2.
    delta_q = 2.

    num_qs = 8  # number of Q rings

    (qd_inds, qd_ring_val, numd_pixels,
     pixeld_list) = diff_roi.roi_rings_step(img_dim, calibrated_center, num_qs,
                                            first_q, delta_q, 2., 2.5, 4., 3.
                                            , 0., 2., 3.)

    ty = np.ones(img_dim).ravel() * np.nan
    ty[pixeld_list] = qd_inds
    ty[np.isnan(ty)] = 0

    numd_pixels_m = (np.bincount(ty.astype(int)))[1:]
    assert_array_equal(numd_pixels, numd_pixels_m)"""

"""qd_ring_val_m = np.array([[2., 4.],
                             [4.4, 6.4],
                             [6.6, 8.6],
                             [9.1, 11.1],
                             [11.5, 13.5],
                             [13.5, 15.5],
                             [16.1, 18.1],
                             [18.3, 20.3]])

    numd_pixels_m = np.array([36, 68, 64, 37, 32, 31, 33, 10])

    qd_inds_m = np.array([2, 2, 2, 2, 2, 3, 3, 3, 4, 2, 1, 1, 1, 1,
                          1, 2, 2, 2, 3, 3, 4, 1, 1, 1, 1, 1, 1, 1,
                          2, 2, 3, 3, 4, 1, 1, 1, 1, 2, 2, 3, 3, 4,
                          1, 1, 1, 1, 2, 2, 3, 3, 4, 1, 1, 1, 1, 2,
                          2, 3, 3, 4, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3,
                          3, 4, 2, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4,
                          2, 2, 2, 2, 2, 3, 3, 3, 4, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 3, 3, 3, 4, 4, 3, 2, 2, 2, 2,
                          2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 3, 3,
                          3, 3, 3, 3, 3, 4, 4, 5, 3, 3, 3, 3, 3, 3,
                          3, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 5, 5, 5,
                          4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
                          6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
                          6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                          6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6,
                          6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                          6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
                          7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                          7, 7, 7, 7, 7, 8, 8, 7, 7, 7, 7, 7, 7, 7,
                          7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                          8, 8, 8, 8])

    #assert_array_equal(qd_inds, np.ravel(qd_inds_m))
    assert_array_almost_equal(qd_ring_val, qd_ring_val_m)
    assert_array_equal(numd_pixels, numd_pixels_m)"""
