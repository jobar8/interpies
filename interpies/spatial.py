# -*- coding: utf-8 -*-
"""
Interpies - a libray for the interpretation of gravity and magnetic data.

spatial.py:
    Functions to manipulate spatial data (grids and points)

@author: Joseph Barraud
Geophysics Labs, 2017
"""
import subprocess

# import numpy
import numpy as np

# import GDAL modules
from osgeo import osr, ogr


def project_points(inputPoints, s_srs=4326, t_srs=23029):
    '''
    Reproject a set of points from one spatial reference to another.
    
    Parameters
    ----------
    s_srs : Integer
        Spatial reference system of the input (source) file. Must be defined as a EPSG code,
        i.e. 23029 for ED50 / UTM Zone 29N
    t_srs : Integer
        Spatial reference system of the output (target) file. Must be defined as a EPSG code,
        i.e. 23029 for ED50 / UTM Zone 29N
        
    Other example: WGS84 = EPSG:4326
    See http://epsg.io/ for all the codes.
    '''
    # input SpatialReference
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(s_srs)
    
    # output SpatialReference
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(t_srs)
    
    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
        
    # Loop through the points 
    outputPoints = []
    for XY in inputPoints:
        point = ogr.CreateGeometryFromWkt("POINT ({} {})".format(*XY))
        point.Transform(coordTrans)
        outputPoints.append([point.GetX(),point.GetY()])
        
    return np.asarray(outputPoints)

    

def extent(xll, yll, cellsize, nrows, ncols, scale=1.,
           registration='gridlines'):
    '''
    Return the extent (xmin,xmax,ymin,ymax) of an image given the coordinates of
    the lower-left corner, the cellsize and the numbers of rows and columns. 
    Registration option controls whether the coordinates indicate the position
    of the centre of the pixels ('pixels') or the corner ('gridlines').
    
    Returns
    -------
    (xmin,xmax,ymin,ymax) 
    '''
    if registration == 'gridlines':
        xmin = xll*scale
        xmax = (xll+(ncols-1)*cellsize)*scale
        ymin = yll * scale
        ymax = (yll + (nrows-1)*cellsize)*scale
    else:
        # This is the complete footprint of the grid considered as an image
        xmin = (xll - cellsize/2.) *scale
        xmax = (xmin + ncols*cellsize) *scale
        ymin = (yll - cellsize/2.) * scale
        ymax = (ymin + nrows*cellsize) *scale
        
    return (xmin,xmax,ymin,ymax)
 

def grid_to_coordinates(xll, yll, cellsize, nrows, ncols):
    '''
    Return vectors of x and y coordinates of the columns and rows of a grid.
    The result does not depend on the registration (gridlines or pixels) of the
    grid.
    
    Returns
    -------
    x, y: vectors of length ncols and nrows, respectively.
    '''
    xmax = xll + ncols*cellsize
    ymax = yll + nrows*cellsize
    
    # 1-D arrays of coordinates (use linspace to avoid errors due to floating point rounding)
    #x = np.arange(xll , xll+ncols*cellsize , cellsize)
    #y = np.arange(ymax , yll - cellsize , -1*cellsize)
    x = np.linspace(xll , xmax , num=ncols, endpoint=False)
    y = np.linspace(yll , ymax , num=nrows, endpoint=False)
    
    return x,y
    

def grid_to_points(xll, yll, cellsize, nrows, ncols, flipy=True):
    '''
    Return x and y coordinates of all the points of a grid.
    The result does not depend on the registration (gridlines or pixels) of the
    grid.
    
    Returns
    -------
    X: numpy array of shape (n,2) where n = nrows * ncols
        A two column array containing the two output vectors.
    '''
    x,y = grid_to_coordinates(xll,yll,cellsize,nrows,ncols)
    
    if flipy:
        y = np.flipud(y)
    xGrid,yGrid = np.meshgrid(x,y)
    
    return np.column_stack((xGrid.flatten(),yGrid.flatten()))
    

def warp(inputFile, outputFile, xsize, ysize, dst_srs, src_srs=None,
         doClip=False, xmin=None, xmax=None, ymin=None, ymax=None, 
         method='bilinear'):
    '''
    Image reprojection and warping utility, with option to clip. 
    This function calls a GDAL executable.
    
    Parameters
    ----------
    inputFile: path to input file
    outputFile: path to output file
    dst_srs: string
        target spatial reference set, for example .prj filename or "EPSG:n"
    xsize: float
        Output cell size in the x direction (in target georeferenced units)
    ysize: float
        Output cell size in the y direction (in target georeferenced units)
    doClip: boolean
        If True, the extent of the reprojected destination are clipped to the
        bounding box defined by (xmin,xmax,ymin,ymax).
    (xmin,xmax,ymin,ymax): floats
        extents of output file (in target SRS) if clipping is required.
    method: string, default is 'bilinear'
        Resampling method to use. Most frequent methods are:
            'near':
                nearest neighbour resampling.
            'bilinear':
                bilinear resampling.
            'cubic':
                cubic resampling.
            'cubicspline':
                cubic spline resampling.
            'lanczos':
                Lanczos windowed sinc resampling.

    '''
    command = 'gdalwarp -overwrite'
    if src_srs is not None:
        command = command + ' -s_srs "{}"'.format(src_srs) 
    command = command + ' -t_srs "{}"'.format(dst_srs) 
    if doClip:
        command = command + ' -te {} {} {} {}'.format(xmin, ymin, xmax, ymax)
    command = command + ' -tr  {} {}'.format(xsize, ysize)
    command = command + ' -r {}'.format(method)
    command = command + ' "{}" "{}"'.format(inputFile, outputFile)
    
    print('GDAL command\n------------\n'+command)
    print('\nOutput\n------')

    # Run the command
    try:
        retMessage = subprocess.check_output(command, shell=False)
        # remove 'b' letter at the beginning of the string
        retMessage = retMessage.decode("utf-8")
    except subprocess.CalledProcessError as err:
        retMessage = "ERROR. GDAL returned code {}.\n{}\n".format(err.returncode, err.output.decode("utf-8"))
    
    return retMessage


