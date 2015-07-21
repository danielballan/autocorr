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

import six
import numpy as np

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from nose.tools import assert_raises

import skxray.core.roi as roi

from skimage import morphology


def test_roi_pixel_values():
    images = morphology.diamond(8)
    # width incompatible with num_rings

    label_array = np.zeros((256, 256))

    # different shapes for the images and labels
    assert_raises(ValueError,
                  lambda: roi.roi_pixel_values(images, label_array))
    # create a label mask
    center = (8., 8.)
    inner_radius = 2.
    width = 1
    spacing = 1
    edges = roi.ring_edges(inner_radius, width, spacing, num_rings=5)
    rings = roi.rings(edges, center, images.shape)

    intensity_data, index = roi.roi_pixel_values(images, rings)
    assert_array_equal(intensity_data[0], ([1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1]))
    assert_array_equal([1, 2, 3, 4, 5], index)


def test_roi_max_counts():
    img_stack1 = np.random.randint(0, 60, size=(50, ) + (50, 50))
    img_stack2 = np.random.randint(0, 60, size=(100, ) + (50, 50))

    img_stack1[0][20, 20] = 60

    samples = (img_stack1, img_stack2)

    label_array = np.zeros((img_stack1[0].shape))

    label_array[img_stack1[0] < 20] = 1
    label_array[img_stack1[0] > 40] = 2

    assert_array_equal(60, roi.roi_max_counts(samples, label_array))


def test_static_test_sets():
    img_stack1 = np.random.randint(0, 60, size=(50, ) + (50, 50))

    label_array = np.zeros((25, 25))

    # different shapes for the images and labels
    assert_raises(ValueError,
                  lambda: roi.mean_intensity(img_stack1, label_array))
    images1 = []
    for i in range(10):
        int_array = np.tril(i*np.ones(50))
        int_array[int_array == 0] = i*100
        images1.append(int_array)

    images2 = []
    for i in range(20):
        int_array = np.triu(i*np.ones(50))
        int_array[int_array == 0] = i*100
        images2.append(int_array)

    samples = np.array((np.asarray(images1), np.asarray(images2)))

    roi_data = np.array(([2, 30, 12, 15], [40, 20, 15, 10]), dtype=np.int64)

    label_array = roi.rectangles(roi_data, shape=(50, 50))

    # test mean_intensity function
    average_intensity, index = roi.mean_intensity(np.asarray(images1),
                                                  label_array)
    # test mean_intensity_sets function
    average_int_sets, index_list = roi.mean_intensity_sets(samples,
                                                           label_array)

    assert_array_equal((list(average_int_sets)[0][:, 0]),
                       [float(x) for x in range(0, 1000, 100)])
    assert_array_equal((list(average_int_sets)[1][:, 0]),
                       [float(x) for x in range(0, 20, 1)])

    assert_array_equal((list(average_int_sets)[0][:, 1]),
                       [float(x) for x in range(0, 10, 1)])
    assert_array_equal((list(average_int_sets)[1][:, 1]),
                       [float(x) for x in range(0, 2000, 100)])

    # test combine_mean_intensity function
    combine_mean_int = roi.combine_mean_intensity(average_int_sets,
                                                  index_list)

    roi_data2 = np.array(([2, 30, 12, 15], [40, 20, 15, 10],
                          [20, 2, 4, 5]), dtype=np.int64)

    label_array2 = roi.rectangles(roi_data2, shape=(50, 50))

    average_int2, index2 = roi.mean_intensity(np.asarray(images1),
                                              label_array2)
    index_list2 = [index_list, index2]

    average_int_sets.append(average_int2)

    # raise ValueError when there is different labels in different image sets
    #  when trying to combine the values
    assert_raises(ValueError,
                  lambda: roi.combine_mean_intensity(average_int_sets,
                                                     index_list2))


def test_circular_average():
    image = np.zeros((12, 12))
    calib_center = (5, 5)
    inner_radius = 1

    edges = roi.ring_edges(inner_radius, width=1, spacing=1, num_rings=2)
    labels = roi.rings(edges, calib_center, image.shape)
    image[labels == 1] = 10
    image[labels == 2] = 10
    bin_cen, ring_avg = roi.circular_average(image, calib_center, nx=6)

    assert_array_almost_equal(bin_cen, [0.70710678, 2.12132034,
                                        3.53553391,  4.94974747,  6.36396103,
                                        7.77817459], decimal=6)
    assert_array_almost_equal(ring_avg, [8., 2.5, 5.55555556, 0.,
                                         0., 0.], decimal=6)


def test_roi_kymograph():
    calib_center = (25, 25)
    inner_radius = 5

    edges = roi.ring_edges(inner_radius, width=2, num_rings=1)
    labels = roi.rings(edges, calib_center, (50, 50))

    images = []
    for i in range(100):
        int_array = i*np.ones(labels.shape)
        images.append(int_array)

    kymograph_data = roi.roi_kymograph(np.asarray(images), labels, num=1)

    assert_almost_equal(kymograph_data[:, 0],  np.arange(100).reshape(100, 1))
