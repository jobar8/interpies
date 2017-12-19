# -*- coding: utf-8 -*-
"""
Interpies - a libray for the interpretation of gravity and magnetic data.

transforms.py:
    Functions for applying derivatives, transforms and filters to grids.

@author: Joseph Barraud
Geophysics Labs, 2017
"""
# Import numpy and scipy
import numpy as np
from scipy import signal
from scipy.ndimage import filters
#from scipy import interpolate
from scipy import ndimage as nd

# Import scikit-learn modules (used for the find_trend function)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

### definitions
pi = np.pi

# kernels for convolution filters
derfilt3 = np.array([-0.5, 0, 0.5], np.float32)
derfilt5 = np.array([1, -8, 0, 8, -1], np.float32)/12  # Five-point stencil vector
prewitt1d = np.array([-1, 0, 1], np.float32)/2

#===============================================================================
# miscellaneous functions
#===============================================================================

def replace_edges(data, ncells=1):
    """Replace the values at the edges of an array with the values calculated 
    with reflection padding. Useful to correct edge effects due to convolution 
    filters.
    """
    return np.pad(data[ncells:-ncells, ncells:-ncells],
                  ncells, mode='reflect', reflect_type='odd')

def fill_nodata(data, invalid=None):
    """Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell. Not very pretty but enough
    for making sure the calculation works.

    Parameters
    ----------
    data:    numpy array of any dimension
    invalid: a binary array of same shape as 'data'. True cells set where data
                value should be replaced.
                If None (default), use: invalid  = np.isnan(data)

    Returns
    -------
    Return a filled array.

    Credits
    -------
    http://stackoverflow.com/a/9262129
    """
    if np.any(np.isnan(data)):
        if invalid is None:
            invalid = np.isnan(data)
        ind = nd.distance_transform_edt(invalid,
                                        return_distances=False,
                                        return_indices=True)
        return data[tuple(ind)]
    else:
        return data

def simple_resample(data, sampling=2):
    '''
    Resample grid by simply picking cells at a given sampling rate.
    The starting point is the lower-left corner of grid so the location
    of the grid is unchanged.
    '''
        
    return np.flipud(np.flipud(data)[::sampling, ::sampling])
    
def find_trend(X, data, degree=1, returnModel=False):
    '''
    Calculate trend in 2D data. The fit is made with a polynomial function of 
    chosen degree. A least-square method is used for the fit.
    '''
    nrows, ncols = data.shape
    # get location of NaNs
    mask = np.isnan(data)
    
    # Fit data with a polynomial surface (or a plane if degree=1)
    model = Pipeline([('poly', PolynomialFeatures(degree)),
                      ('linear', LinearRegression())])
    model.fit(X[~mask.flatten(), :], data[~mask])
    
    # calculate resulting trend
    trend = model.predict(X).reshape((nrows, ncols))
    
    if returnModel:
        return model
    else:
        return trend
   
def stats(data):
    '''
    Return a list of descriptive statistical values.
    '''
    mean = np.nanmean(data)
    sigma = np.nanstd(data)
    minimum = np.nanmin(data)
    maximum = np.nanmax(data)    
    return (mean, sigma, minimum, maximum)

#==============================================================================
# Derivatives with Savitzky-Golay coeficients
#==============================================================================

#-------------------------------------------
# Pre-calculated Savitzky-Golay coeficients
#-------------------------------------------
# John Krumm, Microsoft Research, August 2001
# 
# SavGolSize<m>Order<n>X<i>Y<j> is a filter in row-major order for one polynomial with:
# 	filter size m x m
# 	polynomial order n
# 	filter for coefficient of term (x^i)(y^j)
# These are grouped by size
# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/KRUMM1/SavGol.htm

# Size 2 Order 1
SavGolSize2Order1X0Y0 = np.array([0.25000000,0.25000000,
                                  0.25000000,0.25000000]).reshape((2,2))
SavGolSize2Order1X1Y0 = np.array([-0.50000000,0.50000000,
                                  -0.50000000,0.50000000]).reshape((2,2))
SavGolSize2Order1X0Y1 = np.array([-0.50000000,-0.50000000,
                                  0.50000000,0.50000000]).reshape((2,2))
# Size 3 Order 1
SavGolSize3Order1X0Y0 = np.array([0.11111111,0.11111111,0.11111111,
                                  0.11111111,0.11111111,0.11111111,
                                  0.11111111,0.11111111,0.11111111]).reshape((3,3))
SavGolSize3Order1X1Y0 = np.array([-0.16666667,0.00000000,0.16666667,
                                  -0.16666667,0.00000000,0.16666667,
                                  -0.16666667,0.00000000,0.16666667]).reshape((3,3))
SavGolSize3Order1X0Y1 = np.array([-0.16666667,-0.16666667,-0.16666667,
                                  0.00000000,0.00000000,0.00000000,
                                  0.16666667,0.16666667,0.16666667]).reshape((3,3))
# Size 3 Order 2  ## can be used for quadratic polynomial fit 
SavGolSize3Order2X0Y0 = np.array([-0.11111111,0.22222222,-0.11111111,
                                  0.22222222,0.55555556,0.22222222,
                                  -0.11111111,0.22222222,-0.11111111]).reshape((3,3))
SavGolSize3Order2X1Y0 = np.array([-0.16666667,0.00000000,0.16666667,
                                  -0.16666667,0.00000000,0.16666667,
                                  -0.16666667,0.00000000,0.16666667]).reshape((3,3))
SavGolSize3Order2X2Y0 = np.array([0.16666667,-0.33333333,0.16666667,
                                  0.16666667,-0.33333333,0.16666667,
                                  0.16666667,-0.33333333,0.16666667]).reshape((3,3))
SavGolSize3Order2X0Y1 = np.array([-0.16666667,-0.16666667,-0.16666667,
                                  0.00000000,0.00000000,0.00000000,
                                  0.16666667,0.16666667,0.16666667]).reshape((3,3))
SavGolSize3Order2X1Y1 = np.array([0.25000000,0.00000000,-0.25000000,
                                  0.00000000,0.00000000,0.00000000,
                                  -0.25000000,0.00000000,0.25000000]).reshape((3,3))
SavGolSize3Order2X0Y2 = np.array([0.16666667,0.16666667,0.16666667,
                                  -0.33333333,-0.33333333,-0.33333333,
                                  0.16666667,0.16666667,0.16666667]).reshape((3,3))

#----------------------------------------
def savgol2d(degree, window_size):
    '''
    Calculate coefficients of two-dimensional Savitzky-Golay filters.
    Derived from https://github.com/whatasunnyday/Savitzky-Golay-Filter
    Checked against Krumm's coefficients (see list above).
    
    Parameters
    ----------
    degree: positive integer
        The degree of the polynomial that is fitted to the data points. The 
        greater the degree, the larger the fitting window must be.
    window_size: positive odd integer
        The size of the square window that is used to calculate the fitting 
        polynomial.
        
    Returns
    -------
    coeffs : 2D array of shape (n, `window_size**2`), where n is the number of
        coefficients in a polynomial of degree `degree` with 2 variables (x and y).
        n is equal to (2+d)! / 2d!
        Each of the n rows is a kernel of size `window_size` that can be used
        to smooth 2D data (with the first one) or to calculate derivatives (with
        the others).
    '''
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("Degree of polynomial must be a positive integer")
    if not isinstance(window_size, int) or window_size % 2 == 0 or window_size < 0 :
        raise ValueError("Window size must be a positive odd integer")
    if window_size ** 2 < ((degree + 2) * (degree + 1)) / 2.0:
        raise ValueError("Degree too high for window size")
        
    # create dictionary of exponents
    exps = [ {"x": k - n, "y": n } for k in range(degree + 1) for n in range(k + 1)]
    
    # coordinates of points in window
    n = np.arange(-(window_size - 1)//2, (window_size - 1)//2 + 1,
                  dtype = np.float64)
    dx = np.tile(n, [window_size, 1]).reshape(window_size ** 2, )
    dy = np.repeat(n, window_size)
    
    # array
    A = np.empty((window_size ** 2, len(exps)))
    for i, exp in enumerate(exps):
        A[:,i] = (dx ** exp["x"]) * (dy ** exp["y"])
        
    return np.linalg.pinv(A)
    
#----------------------------------------
# Dictionary to associate types of derivative with Savitzky-Golay coeficients 
# and parameters
sg_dicts = {}
sg_dicts['dx'] = {'index':1,'factor':1,'exponent':1,'flipfunc':np.fliplr}
sg_dicts['dy'] = {'index':2,'factor':-1,'exponent':1,'flipfunc':np.flipud}
sg_dicts['dx2'] = {'index':3,'factor':2,'exponent':2,'flipfunc':np.fliplr}
sg_dicts['dxdy'] = {'index':4,'factor':-1,'exponent':2,'flipfunc':lambda x: np.flipud(np.fliplr(x))}
sg_dicts['dy2'] = {'index':5,'factor':2,'exponent':2,'flipfunc':np.flipud}

def savgol_smooth(data, deg=3, win=5, doEdges=False):
    '''
    Smooth an array by 2D convolution with a Savitzky-Golay (SG) filter.
    It works even if NaNs are present in the data.
    The SG filter is controlled by two parameters, `deg` (degree) and `win` (window 
    size). The amount of smoothing will increase with `win` and decrease with
    `deg`.
    
    Parameters
    ----------
    data: 2D array
        Input data
    deg: positive integer, default 3
        The degree of the Savitzky-Golay filter. The greater the degree, the 
        larger the fitting window must be.
    win: positive odd integer, default 5
        The size of the fitting window that is used to calculate the SG 
        coefficients.
    doEdges: boolean, default True
        Replace the values at the edges of the output array with values calculated 
        by reflection padding. Useful to correct bad edge effects.
        
    '''
    # retrieve Savitzky-Golay coeficients and make kernel
    sg_coeffs = savgol2d(deg,win)
    sg_kernel = sg_coeffs[0].reshape((win,win))
    
    # calculate filtered result by convolution
    convResult = signal.convolve2d(data,sg_kernel,mode='same',
                                   boundary='symm')
    
    # fill edges
    if doEdges:
        convResult = replace_edges(convResult, (win-1)//2)
    
    return convResult
    

def savgol_deriv(data, cellsize, direction='dx', deg=3, win=5, doEdges=True):
    '''
    Calculate horizontal derivatives by convolution with a Savitzky-Golay (SG)
    filter. It works even if NaNs are present in the data.
    
    Parameters
    ----------
    data : 2D array
        Input array
    cellsize: float
        Size of grid cells. Dimensions are assumed to be identical in both the 
        x and y directions.
    direction : {'dx','dy','dx2','dxdy','dy2'}, optional
        Type of derivative. Default is 'dx', first horizontal derivative in the
        x direction. The x axis is "West to East", i.e. along rows of the array.
        The y axis is "South to North", i.e. along columns of the array.
    deg: positive integer, default 3
        The degree of the Savitzky-Golay filter. The greater the degree, the 
        larger the fitting window must be.
    win: positive odd integer, default 5
        The size of the fitting window that is used to calculate the SG 
        coefficients.
    doEdges: boolean, default True
        Replace the values at the edges of the output array with values calculated 
        by reflection padding. Useful to correct bad edge effects.
    '''
    sg_dict = sg_dicts[direction]
    index = sg_dict['index']
    factor = sg_dict['factor']
    exponent = sg_dict['exponent']
    flipfunc = sg_dict['flipfunc']

    # retrieve Savitzky-Golay coeficients and make kernel
    sg_coeffs = savgol2d(deg, win)
    sg_kernel = flipfunc(sg_coeffs[index].reshape((win, win)))  # flip for convolution
    
    # calculate derivative by convolution
    convResult = factor*signal.convolve2d(data, sg_kernel, mode='same',
                                          boundary='symm')/cellsize**exponent
    # fill edges
    if doEdges:
        convResult = replace_edges(convResult, (win-1)//2)
    
    return convResult

#==============================================================================
# fs_deriv - 5-Tap and 7-tap 1st and 2nd discrete derivatives
#==============================================================================
# ** Adapted from Matlab code by Peter Kovesi **
#
# These functions compute 1st and 2nd derivatives of an image using 
# coefficients given by Farid and Simoncelli (2004). The results are significantly
# more accurate than MATLAB's GRADIENT function on edges that are at angles
# other than vertical or horizontal. This in turn improves gradient orientation
# estimation enormously.  If you are after extreme accuracy try using the 7-tap
# coefficients.
#
# Reference: Hany Farid and Eero Simoncelli "Differentiation of Discrete
# Multi-Dimensional Signals" IEEE Trans. Image Processing. 13(4): 496-508 (2004)
#
# Copyright (c) 2010 Peter Kovesi
# http://www.peterkovesi.com/matlabfns/index.html
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# The Software is provided "as is", without warranty of any kind.
# April 2010

def _conv1(a, h):
    return np.convolve(a, h, mode='same')


def _conv2(h1, h2, A):
    '''
    Performs a 1D convolution down the columns using h1 then a 1D
    convolution along the rows using h2. 
    '''
    result = np.apply_along_axis(_conv1, 0, A, h1)
    result = np.apply_along_axis(_conv1, 1, result, h2)
    return result

def fs_coefficients(tap=5, direction='dx'):
    '''
    This function returns the 5-tap or 7-tap coefficients given by Farid 
    and Simoncelli (2004).
    '''
    
    if tap==5:
        if direction in ['dx', 'dy', 'dxdy']:
            # 5-tap 1st derivative coefficients.  These are optimal if you are just
            # seeking the 1st deriavtives.
            p  = np.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659])
            d1 = np.array([0.109604, 0.276691, 0.000000, -0.276691, -0.109604])
            d2 = 0
            
        elif direction in ['dx2', 'dy2', 'dxdy']:  
            # 5-tap 2nd derivative coefficients. The associated 1st derivative
            # coefficients are not quite as optimal as the ones above but are
            # consistent with the 2nd derivative interpolator p and thus are
            # appropriate to use if you are after both 1st and 2nd derivatives.
            p  = np.array([0.030320, 0.249724, 0.439911, 0.249724, 0.030320])
            d1 = np.array([0.104550, 0.292315, 0.000000, -0.292315, -0.104550])
            d2 = np.array([0.232905, 0.002668, -0.471147, 0.002668, 0.232905])
    
    elif tap==7:
        # 7-tap interpolant and 1st and 2nd derivative coefficients
        p  = np.array([0.004711, 0.069321, 0.245410,
                       0.361117, 0.245410, 0.069321, 0.004711])
        d1 = np.array([0.018708, 0.125376, 0.193091,
                       0.000000, -0.193091, -0.125376, -0.018708])
        d2 = np.array([0.055336, 0.137778, -0.056554,
                       -0.273118, -0.056554, 0.137778, 0.055336])
    
    else:
        raise ValueError('The tap value must be either 5 or 7.')
        
    return p, d1, d2


def fs_deriv(data, cellsize, direction='dx', tap=5):
    '''
    Compute 1st or 2nd derivative of an array using the method of Farid and 
    Simoncelli (2004). 
    
    Parameters
    ----------
    data : 2D array
        Input array
    cellsize: float
        Size of grid cells. Dimensions are assumed to be identical in both the 
        x and y directions.
    direction : {'dx','dy','dx2','dxdy','dy2'}, optional
        Type of derivative. Default is 'dx', first horizontal derivative in the
        x direction. The x axis is "West to East", i.e. along rows of the array.
        The y axis is "South to North", i.e. along columns of the array.
    tap: {5, 7}, default 5
        Size of the kernel that is used to calculate the derivative by 
        convolution.
        
    '''
    # Compute coefficients
    p, d1, d2 = fs_coefficients(tap, direction)
    
    # Compute derivatives
    if direction=='dx':
        result = _conv2(p,d1,data)/cellsize
    elif direction=='dy':
        result = -1 * _conv2(d1,p,data)/cellsize # origin is in lower left corner
    elif direction=='dx2':
        result = _conv2(p,d2,data)/cellsize/cellsize
    elif direction=='dy2':
        result = _conv2(d2,p,data)/cellsize/cellsize
    elif direction=='dxdy':
        result = _conv2(p,d1,data)/cellsize
        result = -1 * _conv2(d1,p,result)/cellsize
        
    return result

#==============================================================================
# Fourier functions
#==============================================================================
def getk(nx, ny, dx, dy):
    '''
    Given the size `nx` and `ny` of a FFT and the spacing `dx` and `dy`
    of the space domain grid, this routine returns the spatial
    frequency grid components `kx`, `ky` and `k = sqrt(kx.^2 + ky.^2)`
    
    Makes use of numpy function `fftfreq`.
    
    Returns
    -------
    [kx,ky,k]
    
    '''
    
    # Discrete Fourier Transform sample frequencies
    kx = 2*np.pi*np.fft.fftfreq(nx,dx)
    ky = 2*np.pi*np.fft.fftfreq(ny,dy)
    
    # Create matrices for 2D case
    kx = np.tile(kx,(ny,1))
    ky = np.tile(ky,(nx,1)).T
    
    # calculate k
    k=np.sqrt(kx**2+ky**2)
    
    return [kx,ky,k]


def next_pow2(x):
    '''
    n = up_to_pow2(x)
    return the nearest power of 2 going upwards.
    '''
    return int(2.**np.ceil(np.log(x)/np.log(2)))


# Padding functions
def pad_next_pow2(data, mode='reflect', reflect_type='odd', smooth=False,
                  end_values=0):
    '''
    Pad to a square grid with 2**n number of cells in each dimension, 
    with 2**n being the next power of 2 relative to the size of the input array.
    Use numpy padding function (same mode, reflect_type_type and end_values 
    arguments).
    
    Parameters
    ----------
    data: 2D array
        Input data.
        
    mode : {'reflect', 'linear_ramp'}, optional
        Mode used by secondary padding after tiling. See numpy pad function for
        more information.
        
    reflect_type : {'even', 'odd'}, optional
        Used in 'reflect' mode. The 'odd' style is the default with the extented 
        part of the array created by subtracting the reflected values from 
        two times the edge value. For the 'even' style, the reflection is
        unaltered around the edge value.
        
    smooth : boolean, optional
        option to apply a moving average smoothing function over 
        the edge of the grid.
        default: False
             
    Notes
    -----
    See https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    
    '''
    nrows,ncols = data.shape
    nmax = max([nrows,ncols])
    npts = next_pow2(nmax)  # next 2^n number
    cdiff = (npts - ncols) // 2
    rdiff = (npts - nrows) // 2
    
    # if (npts-nrows) is odd, add 1 row on the bottom side
    r_remainder = np.mod((npts - nrows),2) 
    # if (npts-ncols) is odd, add 1 column on the right-hand side
    c_remainder = np.mod((npts - ncols),2) 
    
    # apply padding
    if mode in ['reflect','symmetric']:
        padded = np.pad(data, ((rdiff, rdiff+r_remainder),(cdiff,cdiff+c_remainder)),
                        mode=mode,reflect_type=reflect_type)
    else:
        padded = np.pad(data, ((rdiff, rdiff+r_remainder),(cdiff,cdiff+c_remainder)),
                        mode=mode,end_values=(end_values,))
        
    if smooth:
        for i in range(-2,3):
            padded[:,cdiff+i] = smoothing_average(padded, cdiff+i, axis='cols')
            padded[:,ncols-1+cdiff+i] = smoothing_average(padded, 
                                                  ncols-1+cdiff+i, axis='cols')
            padded[rdiff+i,:] = smoothing_average(padded, rdiff+i, axis='rows')
            padded[nrows-1+rdiff+i,:] = smoothing_average(padded, 
                                                  nrows-1+rdiff+i, axis='rows')
            
    return padded

def pad_full(data, mode='reflect', reflect_type='odd'):
    '''
    Combine tiling and padding.
    Extend an array first by tiling symmetrical copies of the input
    to a 3x3 array (in reflect mode) then pad with a linear ramp or by reflection 
    to the next power of 2.
    
    Parameters
    ----------
    data: 2D array
        Input data
        
    mode : {'reflect', 'linear_ramp'}, optional
        Mode used by secondary padding after tiling. See numpy pad function for
        more information.
        
    reflect_type : {'even', 'odd'}, optional
        Used in 'reflect' mode. The 'odd' style is the default with the extented 
        part of the array created by subtracting the reflected values from 
        two times the edge value. For the 'even' style, the reflection is
        unaltered around the edge value. 
    
    See also
    --------
    Numpy pad :
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    '''
    nrows, ncols = data.shape
    
    # first 3x3 padding
    data_pad = np.pad(data, ((nrows,nrows), (ncols,ncols)), mode='reflect',
                      reflect_type=reflect_type) 
    
    # additional padding to size = next power of 2
    if mode == 'reflect':
        data_pad = pad_next_pow2(data_pad, mode='reflect', 
                                 reflect_type=reflect_type) 
    else:
        data_pad = pad_next_pow2(data_pad, mode='linear_ramp',
                                 end_values=int(data_pad.mean()))  # linear ramp 
    
    return data_pad

def pad_3x3(data, mode='reflect', reflect_type='odd'):
    '''
    Extend a matrix by tiling symmetrical copies of the input
    Return a 3*nrows x 3*ncols array 
    '''
    nrows, ncols = data.shape
    
    # 3x3 padding
    if mode == 'reflect':
        data_pad = np.pad(data, ((nrows,nrows), (ncols,ncols)), mode=mode,
                          reflect_type=reflect_type)  
    else:
        data_pad = np.pad(data, ((nrows,nrows), (ncols,ncols)), mode='linear_ramp',
                          end_values=int(np.nanmean(data)))  # linear ramp 
    
    return data_pad
    
def unpad_next_pow2(data, nrows, ncols):
    '''
    Retrieve the original array after padding with upPow2_pad.
    (nrows, ncols) is the shape of the array before padding.
    '''
    nmax = max([nrows,ncols])
    npts = next_pow2(nmax)
    cdiff = ((npts - ncols) // 2)
    rdiff = ((npts - nrows) // 2)
    return data[rdiff:nrows + rdiff,cdiff:ncols + cdiff]

       
def unpad_3x3(data):
    '''
    Retrieve the original matrix that was padded with 3x3 reflection padding
    '''
    return np.hsplit(np.vsplit(data, 3)[1], 3)[1]


def unpad_full(data, nrows, ncols):
    '''
    Retrieve the original matrix that was padded with pad_full reflection padding.
    (nrows, ncols) is the shape of the array before padding.
    '''
    data_unpad = unpad_next_pow2(data, 3*nrows, 3*ncols)
    
    return unpad_3x3(data_unpad) # remove 3x3 padding


# put everything together
def fourier_transform(data, cellsize, trans='dx', order=1, doEdges=True, ncells=2, 
                      padding='full', mode='reflect', reflect_type='odd', 
                      eps=1e-6, z=500):
    '''
    Calculate transforms in the frequency domain.
    
    Parameters
    ----------
    data : 2D array
        Input array
    cellsize: float
        Size of grid cells. Dimensions are assumed to be identical in both the 
        x and y directions.
    trans: string
        One of the following string values:
            'dx': horizontal derivative along the x-axis
            'dy': horizontal derivative along the y-axis
            'dxdy': horizontal derivatives along the x-axis and y-axis
            'dz': vertical derivative
            'vi': vertical integral
            'upcont': upward continuation
    order: float, default: 1
        The order of differentiation or integration
    doEdges: boolean, default True
        Replace the values at the edges of the output array with values calculated 
        by reflection padding. Useful to correct bad edge effects.
    ncells: int, default: 2
        Number of cells at the edges of the output grid that are replaced using 
        padding if the `doEdges` option is True.
    padding: string
        Type of padding to apply to the input grid before the Fourier calculation.
        Can be one of the following options:
            'full': initial 3x3 padding (reflect) + ramp or reflection to next power of 2
            '3x3': The entire array is duplicated and tiled in a 3x3 pattern 
                with the original array in the middle.
            'pow2': the size of the array is increased by padding to the next 
                power of 2.
    mode: string, default: 'reflect'
        Option for padding the input array. 
            'reflect': Pads with the reflection of the array
            'linear_ramp': Pads with a linear ramp between the array edge value
                and the mean value of the array.
    reflect_type: string, default: 'odd'
        Used in reflection padding. Can be 'even' or 'odd'. See numpy function pad.
    eps: float
        Small number to replace zeros in frequency components k with when 
        the vertical integral is calculated.
    z: float
        Height used for upward continuation. Default is 500 m.
        
    '''
    nrows,ncols = data.shape
    # save array mask before calculation
    mask = np.isnan(data)
    
    # Apply padding
    padding = padding.lower()
    if padding == 'full':
        # initial 3x3 padding (reflect) + ramp or reflection to next power of 2
        data_pad = pad_full(fill_nodata(data), mode=mode, reflect_type=reflect_type)
    elif padding == '3x3':
        # 3x3 reflection padding
        data_pad = pad_3x3(fill_nodata(data), mode=mode, reflect_type=reflect_type)
    elif padding == 'pow2':
        # ramp or reflection to next power of 2
        data_pad = pad_next_pow2(fill_nodata(data), mode=mode, reflect_type=reflect_type,
                           smooth=True, end_values=int(np.nanmean(data)))
    else:
        # no padding
        data_pad = fill_nodata(data)
    
    # Calculate the k matrix
    (ny,nx) = data_pad.shape
    [kx,ky,k] = getk(nx, ny, cellsize, cellsize)    
    
    # Apply transformation on padded data
    trans = trans.lower()
    if trans == 'dx':
        fouTrans = np.real(np.fft.ifft2(np.fft.fft2(data_pad)*(1j*kx)**order))
    elif trans == 'dy':
        fouTrans = np.real(np.fft.ifft2(np.fft.fft2(data_pad)*(1j*ky)**order))
    elif trans == 'dxdy':
        fouTrans = np.real(np.fft.ifft2(
                            (np.fft.fft2(data_pad)*(1j*ky)**order)*(1j*kx)**order))
    elif trans == 'dz':
        fouTrans = np.real(np.fft.ifft2(np.fft.fft2(data_pad)*k**order))
    elif trans == 'vi':
        # remove zeros in k to avoid division by zero error
        k[k==0] = eps
        fouTrans = np.real(np.fft.ifft2(np.fft.fft2(data_pad)*k**(-1*order)))
        fouTrans = fouTrans - np.mean(fouTrans)
    elif trans == 'upcont':
        fouTrans = np.real(np.fft.ifft2(np.fft.fft2(data_pad)*(np.exp(-z*k))))
    
    # remove padding
    if padding == 'full':
        fouTrans = unpad_full(fouTrans, nrows, ncols)
    elif padding == '3x3':
        fouTrans = unpad_3x3(fouTrans)
    elif padding == 'pow2':
        fouTrans = unpad_next_pow2(fouTrans, nrows, ncols)
    
    # fill edges
    if doEdges:
        fouTrans = replace_edges(fouTrans, ncells)
    
    # re-apply the mask
    fouTrans[mask] = np.nan

    return fouTrans

#===============================================================================
# ISVD (vertical derivative)
#===============================================================================
def isvd(data, cellsize, method='SG', order=1, deg=4, win=5, fs_tap=5,
         doEdges=True, **kwargs):
    ''' Vertical derivatives with the ISVD (integrated second
    vertical derivative) method.
    
    Parameters
    ----------
    data: 2D array
        Input data
    cellsize: float
        Size of grid cells. Dimensions are assumed to be identical in both the 
        x and y directions.
    method: {'SG, 'FS', 'fourier'}, optional
        The method to use for the calculation of the second horizontal 
        derivatives. The three options are:
            - 'SG': Savitzky-Golay method
            - 'FS': Farid and Simoncelli method
            - 'fourier': fourier method
    order: scalar, optional, default: 1
        Order of differentiation. Must be either 1 or 2. If 1, then vertical 
        integration is first applied to the data.
    deg: positive integer, default 4
        The degree of the Savitzky-Golay filter if the SG method is used.
    win: positive odd integer, default 5
        The size of the fitting window that is used to calculate the SG 
        coefficients.
    fs_tap: {5, 7}, default 5
        Size of the kernel that is used to calculate the derivatives with the
        FS method.
    doEdges: boolean, default True
        Replace the values at the edges of the output array with values calculated 
        by reflection padding. Useful to correct bad edge effects.
    kwargs : other keywords
        Options to pass to the fourier transform.
    
    Reference
    ---------
        Fedi, M., Florio, G., 2001. Detection of potential fields source boundaries
        by enhanced horizontal derivative method. Geophys. Prospect. 49, 40–58.
    '''
    if order not in [1, 2]:
        raise ValueError('Order must be 1 or 2.')
        
    # save array mask before calculation
    mask = np.isnan(data)
    
    # fill no data areas (unchanged if no null cells)
    data = fill_nodata(data)
    
    if order==1:
        # vertical integral
        data = fourier_transform(data, cellsize, trans='vi', order=1)
        # smoothing
        if kwargs:
            data = gauss(data, kwargs['sigma'])
        
    # second derivatives
    if method == 'SG':
        data_dx2 = savgol_deriv(data, cellsize, direction='dx2', deg=deg,
                                         win=win, doEdges=doEdges)
        data_dy2 = savgol_deriv(data, cellsize, direction='dy2', deg=deg,
                                         win=win, doEdges=doEdges)
        
    elif method == 'FS':
        data_dx2 = fs_deriv(data, cellsize, direction='dx2', tap=fs_tap)
        data_dy2 = fs_deriv(data, cellsize, direction='dy2', tap=fs_tap)
        
    elif method == 'fourier':
        data_dx2 = fourier_transform(data, cellsize, trans='dx', order=2, **kwargs)
        data_dy2 = fourier_transform(data, cellsize, trans='dy', order=2, **kwargs)
        
    # return DZ using the Laplace equation
    data_dz = -1*(data_dx2 + data_dy2)
    
    # fill edges
    if doEdges:
        data_dz = replace_edges(data_dz, (win-1)//2)
        
    # re-apply mask
    data_dz[mask] = np.nan
    
    return data_dz


#===============================================================================
# Various filters
#===============================================================================
def gauss(data, sigma=1):
    
    return filters.gaussian_filter(data, sigma)


def smoothing_average(V, i, axis='cols'):
    if axis == 'cols':
        Vs = (V[:,i-2]+V[:,i-1]+V[:,i]+V[:,i+1]+V[:,i+2])/5.
    else:
        Vs = (V[i-2,:]+V[i-1,:]+V[i,:]+V[i+1,:]+V[i+2,:])/5.
    return Vs


def laplacian(data, cellsize):
    conv_filter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    convResult = signal.convolve2d(data, conv_filter, 
                                   mode='valid',boundary='symm')/cellsize
    
    return convResult

