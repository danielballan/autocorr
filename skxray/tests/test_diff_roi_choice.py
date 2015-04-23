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

import skxray.diff_roi_choice as roi
import skxray.correlation as corr
import skxray.core as core

from skxray.testing.decorators import known_fail_if
import numpy.testing as npt


def test_roi_rectangles():
    num_rois = 3
    detector_size = (15, 26)
    roi_data = np.array(([2, 2, 6, 3], [6, 7, 8, 5], [8, 18, 5, 10]),
                        dtype=np.int64)

    all_roi_inds = roi.rectangles(num_rois, roi_data, detector_size)

    roi_inds, pixel_list = corr.extract_label_indices(all_roi_inds)

    ty = np.zeros(detector_size).ravel()
    ty[pixel_list] = roi_inds
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


def test_roi_rings():
    calibrated_center = (6., 4.)
    img_dim = (20, 25)
    first_q = 2
    delta_q = 3
    num_qs = 7  # number of Q rings

    all_roi_inds = roi.rings(img_dim, calibrated_center, num_qs,
                             first_q, delta_q)
    ring_vals = roi.rings_edges(num_qs, first_q, delta_q)

    q_inds, pixel_list = corr.extract_label_indices(all_roi_inds)

    num_pixels = np.bincount(q_inds)[1:]

    ring_vals = roi.rings_edges(num_qs, first_q, delta_q)

    # Edge values of each rings
    ring_edges = []

    for i in range(num_qs):
        if i < num_qs:
            ring_edges.append(ring_vals[i])
            ring_edges.append(ring_vals[i + 1])

    ring_edges = np.asarray(ring_edges)
    ring_edges = ring_edges.reshape(num_qs, 2)

    # check the rings edge values
    q_ring_val_m = np.array([[2, 5], [5, 8], [8, 11], [11, 14], [14, 17],
                             [17, 20], [20, 23]])

    assert_array_almost_equal(q_ring_val_m, ring_edges)

    # check the pixel_list and q_inds and num_pixels
    _helper_check(pixel_list, q_inds, num_pixels, ring_edges,
                  calibrated_center, img_dim, num_qs)


def test_roi_rings_step():
    calibrated_center = (4., 6.)
    img_dim = (20, 25)
    first_q = 2.5
    delta_q = 2

    # using a step for the Q rings
    num_qs = 6  # number of Q rings
    step_q = (1, )  # step value between each Q ring

    all_roi_inds = roi.rings_step(img_dim, calibrated_center, num_qs,
                                  first_q, delta_q, *step_q)

    ring_vals = roi.rings_step_edges(num_qs, first_q, delta_q, *step_q)
    q_ring_val = roi.process_ring_edges(ring_vals)

    q_inds, pixel_list = corr.extract_label_indices(all_roi_inds)

    # get the number of pixels in each Q ring
    num_pixels = np.bincount(q_inds)[1:]

    # check the ring edge values
    q_ring_val_m = np.array([[2.5, 4.5], [5.5, 7.5], [8.5, 10.5],
                             [11.5, 13.5], [14.5, 16.5], [17.5, 19.5]])

    assert_almost_equal(q_ring_val, q_ring_val_m)

    # check the pixel_list and q_inds and num_pixels
    _helper_check(pixel_list, q_inds, num_pixels, q_ring_val,
                  calibrated_center, img_dim, num_qs)


def test_roi_rings_diff_steps():
    calibrated_center = (10., 4.)
    img_dim = (45, 25)
    first_q = 2.
    delta_q = 2.

    num_qs = 8  # number of Q rings

    step_q = (2., 2.5, 4., 3., 0., 2.5, 3.)
    all_roi_inds = roi.rings_step(img_dim, calibrated_center, num_qs,
                                  first_q, delta_q, *step_q)

    ring_vals = roi.rings_step_edges(num_qs, first_q, delta_q, *step_q)

    q_ring_val = roi.process_ring_edges(ring_vals)

    q_inds, pixel_list = corr.extract_label_indices(all_roi_inds)

    # get the number of pixels in each Q ring
    num_pixels = np.bincount(q_inds)[1:]

    # check the edge values of the rings
    q_ring_val_m = np.array([[2., 4.], [6., 8.], [10.5, 12.5], [16.5, 18.5],
                             [21.5, 23.5], [23.5, 25.5], [28.0, 30.0],
                             [33.0, 35.0]])

    assert_array_almost_equal(q_ring_val, q_ring_val_m)

    # check the pixel_list and q_inds and num_pixels
    _helper_check(pixel_list, q_inds, num_pixels, q_ring_val,
                  calibrated_center, img_dim, num_qs)


def _helper_check(pixel_list, inds, num_pix, q_ring_val, calib_center,
                  img_dim, num_qs):
    # recreate the indices using pixel_list and inds values
    ty = np.zeros(img_dim).ravel()
    ty[pixel_list] = inds
    data = ty.reshape(img_dim[0], img_dim[1])

    # get the grid values from the center
    grid_values = core.pixel_to_radius(img_dim, calib_center)

    # get the indices into a grid
    zero_grid = np.zeros((img_dim[0], img_dim[1]))
    for r in range(num_qs):
        vl = (q_ring_val[r][0] <= grid_values) & (grid_values
                                                  < q_ring_val[r][1])
        zero_grid[vl] = r + 1

    # check the num_pixels
    num_pixels = []
    for r in range(num_qs):
        num_pixels.append(int((np.histogramdd(np.ravel(grid_values), bins=1,
                                              range=[[q_ring_val[r][0],
                                                      (q_ring_val[r][1]
                                                       - 0.000001)]]))[0][0]))
    assert_array_equal(zero_grid, data)
    assert_array_equal(num_pix, num_pixels)


if __name__ == " __main__":
    test_roi_rings()
    test_roi_rings_step()
    test_roi_rings_diff_steps()