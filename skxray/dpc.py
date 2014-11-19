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
This module is for Differential Phase Contrast (DPC) imaging based on
Fourier shift fitting
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.optimize import minimize


def image_reduction(im, roi=None, bad_pixels=None):
    """ 
    Sum the image data along one dimension
        
    Parameters
    ----------
    im : 2-D numpy array
        store the image data
    
    roi : tuple
        store the top-left and bottom-right coordinates of an rectangular ROI
        roi = (11, 22, 33, 44) --> (11, 21) - (33, 43)
        
    bad_pixels : list
        store the coordinates of bad pixels
        [(1, 5), (2, 6)] --> 2 bad pixels --> (1, 5) and (2, 6)
    
    Returns
    ----------
    xline : 1-D numpu array
        the sum of the image data along x direction
        
    yline : 1-D numpy array
        the sum of the image data along y direction
        
    """
      
    if bad_pixels is not None:
        for x, y in bad_pixels:
            try:
                im[x, y] = 0
            except IndexError:
                print("Bad pixel indexes are out of range.")
                
    if roi is not None:
        x1, y1, x2, y2 = roi
        try:
            im = im[x1:x2, y1:y2]
        except IndexError:
            print("The ROI is out of range.")
        
    xline = np.sum(im, axis=0)
    yline = np.sum(im, axis=1)
        
    return xline, yline


def _rss(v, xdata, ydata):
    """ 
    Internal function used by fit()
    Cost function to be minimized in nonlinear fitting
    
    Parameters
    ----------
    v : list
        store the fitting value
        v[0], intensity attenuation
        v[1], phase gradient along x or y direction
    
    xdata : 1-D complex numpy array
        auxiliary data in nonlinear fitting
        returning result of ifft1D()
    
    ydata : 1-D complex numpy array
        auxiliary data in nonlinear fitting
        returning result of ifft1D()
        
    Returns
    --------
    residue : float
        residue value
    
    """
    
    length = len(xdata)
    beta = 1j * (np.linspace(-(length-1)//2, (length-1)//2, length))
    
    fitted_curve = xdata * v[0] * np.exp(v[1] * beta)
    residue = np.sum(np.abs(ydata - fitted_curve) ** 2)
    
    return residue



def dpc_fit(ref_f, f, start_point=[1, 0], solver='Nelder-Mead', tol=1e-8, 
        max_iters=2000):
    """ 
    Nonlinear fitting for 2 points 
    
    Parameters
    ----------
    ref_f : 1-D numpy array
        One of the two arrays used for nonlinear fitting
     
    f : 1-D numpy array
        One of the two arrays used for nonlinear fitting

    start_point : 2-element list
        start_point[0], start-searching point for the intensity attenuation
        start_point[1], start-searching point for the phase gradient
    
    solver : string
        method to solve the nonlinear fitting problem
    
    tol : float
        termination criteria of nonlinear fitting
        
    max_iters : integer
        maximum iterations of nonlinear fitting
        
    Returns:
    ----------
    a : float
        fitting result: intensity attenuation

    g : float
        fitting result: phase gradient
    
    See Also:
    ---------    
    _rss() : function
        objective function to be minimized in the fitting algorithm
    
    """
        
    res = minimize(_rss, start_point, args=(ref_f, f), method=solver, tol=tol, 
                   options=dict(maxiter=max_iters))
                    
    vx = res.x
    a = vx[0]
    g = vx[1]
        
    return a, g



def recon(gx, gy, dx=0.1, dy=0.1, pad=1, w=1.):
    """ 
    Reconstruct the final phase image 

    Parameters
    ----------
    gx : 2-D numpy array
        phase gradient along x direction
    
    gy : 2-D numpy array
        phase gradient along y direction
    
    dx : float
        scanning step size in x direction (in micro-meter)
        
    dy : float
        scanning step size in y direction (in micro-meter)
    
    pad : float
        padding parameter
        default value, pad = 1 --> no padding
                    p p p
        pad = 3 --> p v p
                    p p p
                    
    w : float
        weighting parameter for the phase gradient along x and y direction when
        constructing the final phase image
        
    Returns
    ----------
    phi : 2-D numpy array
        final phase image
        
    References
    ----------
    [1] Yan, Hanfei, Yong S. Chu, Jorg Maser, Evgeny Nazaretski, Jungdae Kim,
    Hyon Chol Kang, Jeffrey J. Lombardo, and Wilson KS Chiu, "Quantitative
    x-ray phase imaging at the nanoscale by multilayer Laue lenses," Scientific 
    reports 3 (2013).
        
    """
    
    rows, cols = gx.shape

    gx_padding = np.zeros((pad * rows, pad * cols), dtype='d')
    gy_padding = np.zeros((pad * rows, pad * cols), dtype='d')
    
    gx_padding[(pad // 2) * rows : (pad // 2 + 1) * rows,
               (pad // 2) * cols : (pad // 2 + 1) * cols] = gx
    gy_padding[(pad // 2) * rows : (pad // 2 + 1) * rows, 
               (pad // 2) * cols : (pad // 2 + 1) * cols] = gy
    
    tx = np.fft.fftshift(np.fft.fft2(gx_padding))
    ty = np.fft.fftshift(np.fft.fft2(gy_padding))
    
    c = np.zeros((pad * rows, pad * cols), dtype=complex)
    
    mid_col = pad * cols // 2.0 + 1
    mid_row = pad * rows // 2.0 + 1

    ax = 2 * np.pi * (np.arange(pad * cols) + 1 - mid_col) / (pad * cols * dx)
    ay = 2 * np.pi * (np.arange(pad * rows) + 1 - mid_row) / (pad * rows * dy)

    kappax, kappay = np.meshgrid(ax, ay)

    c = -1j * (kappax * tx + w * kappay * ty)

    c = np.ma.masked_values(c, 0)
    c /= (kappax**2 + w * kappay**2)
    c = np.ma.filled(c, 0)

    c = np.fft.ifftshift(c)
    phi_padding = np.fft.ifft2(c)
    phi_padding = -phi_padding.real
    
    phi = phi_padding[(pad // 2) * rows : (pad // 2 + 1) * rows,
                      (pad // 2) * cols : (pad // 2 + 1) * cols]
    
    return phi



def dpc_runner(start_point = [1, 0], pixel_size = 55, focus_to_det = 1.46e6, 
               rows = 121, cols = 121, energy = 19.5, roi = None, pad = 1., 
               w = 1., bad_pixels = None, solver = 'Nelder-Mead', 
               image_size = (61, 91), ref = None, image_sequence = None):
    """
    Controller function to run the whole DPC
    
    Parameters
    ----------
    start_point : 2-element list
        start_point[0], start-searching point for the intensity attenuation
        start_point[1], start-searching point for the phase gradient
        
    pixel_size : integer
        pixel size of the detector
    
    focus_to_det : integer
        focus to detector distance
    
    rows : integer
        number of scanned rows 
    
    cols : integer
        number of scanned columns
    
    energy : float
        energy of the scanning x-ray
    
    roi : tuple
        store the top-left and bottom-right coordinates of an rectangular ROI
        roi = (11, 22, 33, 44) --> (11, 21) - (33, 43)
        
    pad : float
        padding parameter
        default value, pad = 1 --> no padding
                    p p p
        pad = 3 --> p v p
                    p p p
    
    w : float
        weighting parameter for the phase gradient along x and y direction when
        constructing the final phase image
    
    bad_pixels : list
        store the coordinates of bad pixels
        [(1, 5), (2, 6)] --> 2 bad pixels --> (1, 5) and (2, 6)
    
    solver : string
        method to solve the nonlinear fitting problem
    
    image_size : tuple
        image_size[0], the number of rows for each scanned image
        image_size[1], the number of columns for each scanned image
    
    ref : 2-D numpy array
        store the reference image
        
    image_sequence : 3-D numpy array
        store the set of scanned images
        
    Returns
    -------
    phi : 2-D numpy array
        the final reconstructed phase image
    
    """
    
    # Initialize a, gx, gy and phi
    a = np.zeros((rows, cols), dtype='d')
    gx = np.zeros((rows, cols), dtype='d')
    gy = np.zeros((rows, cols), dtype='d')
    phi = np.zeros((rows, cols), dtype='d')

    # Dimension reduction along x and y direction
    refx, refy = image_reduction(ref, roi=roi)

    # 1-D IFFT
    ref_fx = np.fft.fftshift(np.fft.ifft(refx))
    ref_fy = np.fft.fftshift(np.fft.ifft(refy))

    # Same calculation on each diffraction pattern
    for index, im in enumerate(image_sequence):
        i, j = np.unravel_index(index, (rows, cols))
        print(index)
        # Dimension reduction along x and y direction
        imx, imy = image_reduction(im, roi=roi)
                
        # 1-D IFFT
        fx = np.fft.fftshift(np.fft.ifft(imx))
        fy = np.fft.fftshift(np.fft.ifft(imy))
                
        # Nonlinear fitting
        _a, _gx = dpc_fit(ref_fx, fx)
        _a, _gy = dpc_fit(ref_fy, fy)
                            
        # Store one-point intermediate results
        gx[i, j] = _gx
        gy[i, j] = _gy
        a[i, j] = _a
        
    # Scale gx and gy. Not necessary all the time
    lambda_ = 12.4e-4 / energy
    gx *= - len(ref_fx) * pixel_size / (lambda_ * focus_to_det)
    gy *= len(ref_fy) * pixel_size / (lambda_ * focus_to_det)

    # Reconstruct the final phase image
    phi = recon(gx, gy)
    
    return phi
