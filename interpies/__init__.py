"""
Interpies - a libray for the interpretation of gravity and magnetic data.


@author: Joseph Barraud
Geophysics Labs, 2018
"""

__version__ = "0.3.1"

import rasterio
from interpies.grid import Grid, from_dataset

def open(inputFile, crs=None, name=None, nodata_value=None,
         scale_factor=None, **kwargs):
    """Open a dataset using the rasterio open function.
    This returns a grid object, which is basically a 2D array with attributes
    attached to it. In other words, grids are rasters with only one band.
    If the input dataset has several bands, then only the first one is read.

    Parameters
    ----------
    inputFile : path to a raster dataset
        The path may point to a file of any raster format supported by rasterio,
        which in turn can be any format supported by GDAL.
    crs : string, optional
        Coordinate reference system. This is optional as the CRS will normally
        be read by rasterio and the suitable GDAL driver. This can be used in
        case the CRS definition is absent from the raster file (for example
        when the input is a simple XYZ text file).
    nodata_value : float, optional
        No data value. This is optional as the no data value will normally
        be read by rasterio and the suitable GDAL driver.
    scale_factor : float, optional
        The input data will be multiplied by this number is present. The
        scale_factor is sometimes present in netCDF files but unfortunately
        rasterio does not apply the scaling automatically.
    kwargs: additional keywords are passed to the rasterio open function.
    """
    dataset = rasterio.open(inputFile, nodata=nodata_value, **kwargs)

    return from_dataset(dataset, crs=crs, name=name,
                        nodata_value=nodata_value,
                        scale_factor=scale_factor)
