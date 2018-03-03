# -*- coding: utf-8 -*-
"""
Interpies - a libray for the interpretation of gravity and magnetic data.

grid.py:
    Sub-module for the definition of the grid class

@author: Joseph Barraud
Geophysics Labs, 2017
"""
import os.path
import rasterio
import numpy as np

# import local modules
from interpies import transforms, spatial, graphics

#==============================================================================
# Grid class
#==============================================================================
class Grid(object):
    '''
    A class containing a grid object.
    '''
    ### object creation ###
    def __init__(self, data, transform=None, nodata_value=None,
                 name='Unknown', filename='Unknown', mask=None, crs=None, copyFrom=None):
        '''
        Create a new grid object by combining a 2D array (data) and georeferecing
        information (rasterio transform). The transform is an affine transformation
        matrix that maps pixel locations in (row, col) coordinates to (x, y)
        spatial positions. It basically gives the position of the upper left
        corner of the dataset, as well as the cell size.
        '''
        # copy parameters from other grid
        if isinstance(copyFrom, Grid):
            transform = copyFrom.transform
            nodata_value = copyFrom.nodata
            filename = copyFrom.filename
            mask = copyFrom.mask
            crs = copyFrom.crs

        # load data array - the conversion to float allows the creation of masks with NaNs
        self.data = data.astype(float)

        # read provided transform or create default one
        if transform is not None:
            self.transform = transform
        else:
            self.transform = rasterio.transform.from_origin(0, 0, 100, 100)

        # define grid properties
        self.name = name
        self.filename = filename
        self.nrows, self.ncols = data.shape
        self.cellsize = self.transform[0]  # requires rasterio > 1.0
        self.y_cellsize = -self.transform[4]  # requires rasterio > 1.0
        if crs is not None:
            self.crs = crs
        else:
            self.crs = 'Unknown'

        # lower left corner (pixel centre)
        self.xll, self.yll = rasterio.transform.xy(self.transform,
                                                   self.nrows-1, 0,
                                                   offset='center')

        # nodata value
        self.nodata = nodata_value
        # convert nodata values to nans
        if nodata_value is not None:
            self.data[self.data == nodata_value] = np.nan

        # apply mask if present
        if mask is not None:
            self.data[mask] = np.nan

        # save mask
        self.mask = np.isnan(self.data)
        self.saved_mask = self.mask
        self.masked = True

        # calculate extent in matplotlib sense
        # matplotlib extent: `left, right, bottom, top`
        # rasterio bounds: `west, south, east, north`
        w, s, e, n = rasterio.transform.array_bounds(self.nrows,
                                                     self.ncols,
                                                     self.transform)
        self.extent = [w, e, s, n]


    def save(self, outputFile):
        '''
        Write grid data to file. The only format supported at the moment is Geotiff.
        '''
        # create rasterio object
        new_dataset = rasterio.open(outputFile, 'w', driver='GTiff',
                                    height=self.nrows, width=self.ncols,
                                    count=1, dtype=rasterio.dtypes.float64,
                                    crs=self.crs, transform=self.transform)
        # write to file
        new_dataset.write(self.data, 1)
        # close file
        new_dataset.close()
        print('The grid was successfully saved to {}'.format(outputFile))


    def to_fatiando(self):
        '''
        Convert a grid to the fatiando "format" in which the (x,y) coordinates
        and the data are given in separate vectors and array.
        '''
        # get the coordinates of the grid cells
        pts = spatial.grid_to_points(self.xll, self.yll,
                                     self.cellsize, self.nrows, self.ncols, flipy=True)

        return pts[:, 1], pts[:, 0], self.data.flatten(), (self.nrows, self.ncols)


    ### Grid methods
    def clip(self, xmin, xmax, ymin, ymax):
        '''
        Clip grid.
        '''
        rows, cols = rasterio.transform.rowcol(self.transform,
                                               [xmin, xmax], [ymin, ymax])

        data_selection = self.data[rows[1]:rows[0]+1, cols[0]:cols[1]+1]

        # new origin
        new_west, new_north = rasterio.transform.xy(self.transform,
                                                    rows[1], cols[0], offset='ul')
        # new transform
        new_transf = rasterio.transform.from_origin(new_west, new_north,
                                                    self.cellsize, self.cellsize)

        return Grid(data_selection, new_transf, name=self.name+'_clip', nodata_value=self.nodata)


    def resample(self, sampling=2):
        '''
        Resample grid (from lower-left corner).
        The new shape and cellsize of the grid are calculated.
        '''
        new_data = transforms.simple_resample(self.data, sampling)
        new_cellsize = sampling*self.cellsize
        new_height, _ = new_data.shape
        # new origin
        new_west = self.extent[0] # unchanged
        new_north = self.extent[2] + new_height * new_cellsize
        # new transform
        new_transf = rasterio.transform.from_origin(new_west, new_north,
                                                    new_cellsize, new_cellsize)

        return Grid(new_data, new_transf, name=self.name+'_res{}'.format(sampling),
                    nodata_value=self.nodata)


    def reproject(self, outputFile, src_srs=None, dst_srs=None, cellsize=None,
                  doClip=False, xmin=None, xmax=None, ymin=None, ymax=None,
                  method='bilinear'):
        '''
        Project or reproject the data to a new coordinate system.

        Parameters
        ----------
        src_srs : string
            Input coordinate system. Can be specified with a EPSG code, i.e.
            "EPSG:4326" for WGS84.
        dst_srs: string
            Destination coordinate system.
        '''
        if src_srs is None:
            if self.crs == 'Unknown':
                raise ValueError('The coordinate system of the input data '
                                 'is undefined. Please set it up in the Grid or '
                                 'use the src_srs argument.')
            else:
                src_srs = self.crs
        if cellsize is None:
            cellsize = self.cellsize
        # call gdalwarp utility
        retMessage = spatial.warp(self.filename, outputFile, cellsize, cellsize,
                                  dst_srs, src_srs, doClip=doClip,
                                  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                  method=method)
        print(retMessage)

        # open output file and create grid object
        dataset = rasterio.open(outputFile)
        return from_dataset(dataset)


    def scale(self, scale_factor):
        '''
        Multiply data with a number.
        '''
        # return scaled grid
        return Grid(scale_factor * self.data, self.transform,
                    name=self.name+'_scaled', nodata_value=self.nodata)


    def detrend(self, degree=1, sampling=4):
        '''
        Apply detrending to the grid. Trend is calculated with a resampled
        version of the array to limit the influence of extrema and also to
        speed up the process.
        '''
        data = self.data
        nr, nc = self.nrows, self.ncols

        if sampling > 1:
            # resample just for the calculation of the trend
            data = transforms.simple_resample(self.data, sampling)
            nr, nc = data.shape

        # get the coordinates of the (whole) grid points
        Xfull = spatial.grid_to_points(self.xll, self.yll,
                                       self.cellsize, self.nrows, self.ncols, flipy=True)
        # get the coordinates of the resampled grid points
        X = spatial.grid_to_points(self.xll, self.yll,
                                   sampling*self.cellsize, nr, nc, flipy=True)
        # Fit data with a polynomial surface (or a plane if degree=1)
        model = transforms.find_trend(X, data, degree=degree, returnModel=True)
        # calculate resulting trend with all the points
        trend = model.predict(Xfull).reshape((self.nrows, self.ncols))

        # return detrended grid
        return Grid(self.data - trend, self.transform,
                    name=self.name+'_detrend', nodata_value=self.nodata)


    def fill_nodata(self):
        '''Simple filling algorithm to remove NaNs.
        '''
        filled = transforms.fill_nodata(self.data)
        # return filled grid
        return Grid(filled, self.transform,
                    name=self.name+'_filled', nodata_value=self.nodata)


    def apply_mask(self, mask=None, inplace=False):
        '''Mask data with new mask or apply saved mask.
        '''
        if mask is None:
            self.data[self.saved_mask] = np.nan
        else:
            self.data[mask] = np.nan
            self.mask = mask
            self.saved_mask = self.mask

        if not inplace:
            return self


    def info(self):
        '''
        Return information about the properties of the grid and the data.
        '''
        print('\n* Info *')
        print('Grid name: ' + self.name)
        print('Filename: ' + self.filename)
        print('Coordinate reference system: ' + self.crs)
        print('Grid size: {:d} columns x {:d} rows'.format(self.ncols, self.nrows))
        print('Cell size: {:.4g} x {:.4g}'.format(self.cellsize, self.y_cellsize))
        print('Lower left corner (pixel centre): ({:.3f}, {:.3f})'
              .format(self.xll, self.yll))
        print('Grid extent (outer limits): ' +
              'west: {:.3f}, east: {:.3f}, south: {:.3f}, north: {:.3f}'
              .format(*self.extent))
        print('No Data Value: {}'.format(self.nodata))
        nanCells = np.isnan(self.data).sum()
        print('Number of null cells: {} ({:.2f}%)'.format(nanCells,
                                                          100*nanCells/(self.nrows * self.ncols)))
        # stats
        mean, sigma, minimum, maximum = transforms.stats(self.data)
        print('\n* Statistics *')
        print('mean = {}'.format(mean))
        print('sigma = {}'.format(sigma))
        print('min = {}'.format(minimum))
        print('max = {}'.format(maximum))


    ### Graphics
    def show(self, ax=None, cmap='geosoft', cmap_norm='equalize', hs=True,
             zf=10, azdeg=45, altdeg=45, dx=1, dy=1, hs_contrast=1.5, cmap_brightness=1.0,
             blend_mode='alpha', alpha=0.7, contours=False, colorbar=True,
             cb_contours=False, cb_ticks='linear', std_range=1, figsize=(8, 8),
             title=None, **kwargs):
        '''
        Display a grid with optional hillshading and contours.
        The data representation is controlled by the colormap and two types
        of normalisation can be applied to balance an uneven distribution of values.
        Contrary to the standard method available in `plt.imshow`, it is the
        colormap, not the data, that is modified. This allows the true distribution
        of the data to be displayed on the colorbar. The two options are equalisation
        (default) or clipping extremes (autolevels).

        Parameters
        ----------
        source : 2D array or interpies grid object
            Grid to plot. Arrays with NaNs and masked arrays are supported.
        ax : matplotlib Axes instance
            This indicates where to make the plot. Create new figure if absent.
        cmap : string or colormap object
            Colormap or name of the colormap to use to display the array. The default
            is 'geosoft' and corresponds to the blue to pink clra colormap from
            Geosoft Oasis Montaj.
        cmap_norm : string
            Type of normalisation of the colormap.
            Possible values are:
                'equalize' (or 'equalization')
                    Increases contrast by distributing intensities across all the
                    possible colours. The distribution is calculated from the
                    data and applied to the colormap.
                'auto' (or 'autolevels')
                    Stretches the histogram of the colormap so that dark colours become
                    darker and the bright colours become brighter. The extreme values
                    are calculated with percentiles: min_percent (defaults to 2%) and
                    max_percent (defaults to 98%).
                'none' or any other value
                    The colormap is not normalised. The data can still be normalised
                    in the usual way using the 'norm' keyword argument and a
                    Normalization instance defined in matplotlib.colors().
        hs : boolean
            If True, the array is displayed in colours over a grey hillshaded version
            of the data.
        zf : float
            Vertical exaggeration (Z factor) for hillshading.
        azdeg : float
            The azimuth (0-360, degrees clockwise from North) of the light source.
        altdeg : float
            The altitude (0-90, degrees up from horizontal) of the light source.
        dx : float, optional
            cell size in the x direction
        dy : float, optional
            cell size in the y direction
        hs_contrast : float
            Increase or decrease the contrast of the hillshade. This is directly
            passed to the fraction argument of the matplotlib hillshade function.
        cmap_brightness : float
            Increase or decrease the brightness of the image by adjusting the
            gamma of the colorbar. Default value is 1.0 meaning no effect. Values
            greater than 1.0 make the image brighter, less than 1.0 darker.
            Useful when the presence of the hillshade makes the result a little
            too dark.
        blend_mode :  {'alpha', 'hsv', 'overlay', 'soft'}
            The type of blending used to combine the colormapped data values with the
            illumination intensity. Default is 'alpha' and the effect is controlled
            by the alpha parameter.
        alpha : float
            Controls the transparency of the data overlaid over the hillshade.
            1.0 is fully opaque while 0.0 is fully transparent.
        contours : Boolean or integer
            If True, add contours to the map, the number of them being the default value, i.e. 32.
            If an integer is given instead, use this value as the number of contours
            levels.
        colorbar : Boolean
            If True, draw a colorbar on the right-hand side of the map. The colorbar
            shows the distribution of colors, as modified by the normalization algorithm.
        cb_ticks : string
            If left as default ('linear') the ticks and labels on the colorbar are
            spaced linearly in the standard way. Otherwise (any other keyword, for example
            'stats'), the mean and two ticks at + and - std_range*(standard deviation)
            are shown instead.
                std_range : integer (default is 1)
                    Extent of the range from the mean as a multiple of the standard deviation.
        cb_contours : Boolean
            Add lines corresponding to contours on the colorbar.
        figsize: tuple
            Dimensions of the figure: width, height in inches.
            If not provided, the default is (8, 8).
        title: string
            String to display as a title above the plot. If the source is a grid
            object, the title is taken by default from the name of the grid.
        kwargs : other optional arguments
            Can be used to pass other arguments to `plt.imshow()`, such as 'origin'
            and 'extent', or for the colorbar('shrink'), or the title
            ('fontweight' and 'fontsize').

        Returns
        -------
        ax : Matplotlib Axes instance.

        Notes
        -----
        This function exploits the hillshading capabilities implemented in
        matplotlib.colors.LightSource. A new blending mode is added (alpha compositing,
        see https://en.wikipedia.org/wiki/Alpha_compositing).
        '''
        if 'origin' in kwargs:
            return graphics.imshow_hs(self, ax=ax, cmap=cmap, cmap_norm=cmap_norm,
                                      hs=hs, zf=zf, azdeg=azdeg, altdeg=altdeg,
                                      dx=dx, dy=dy, hs_contrast=hs_contrast,
                                      cmap_brightness=cmap_brightness, blend_mode=blend_mode,
                                      alpha=alpha, contours=contours,
                                      colorbar=colorbar, cb_contours=cb_contours,
                                      cb_ticks=cb_ticks, std_range=std_range,
                                      figsize=figsize, title=title, **kwargs)

        # set origin to ensure that both grid and contours get the same origin
        return graphics.imshow_hs(self, ax=ax, cmap=cmap, cmap_norm=cmap_norm,
                                  hs=hs, zf=zf, azdeg=azdeg, altdeg=altdeg,
                                  dx=dx, dy=dy, hs_contrast=hs_contrast,
                                  cmap_brightness=cmap_brightness, blend_mode=blend_mode,
                                  alpha=alpha, contours=contours,
                                  colorbar=colorbar, cb_contours=cb_contours,
                                  cb_ticks=cb_ticks, std_range=std_range,
                                  figsize=figsize, title=title,
                                  origin='upper', **kwargs)


    def save_image(self, output_file, scale=1, cmap='geosoft', cmap_norm='equalize', hs=True,
                   zf=10, azdeg=45, altdeg=45, dx=1, dy=1, hs_contrast=1.5, cmap_brightness=1.0,
                   blend_mode='alpha', alpha=0.7, contours=False, **kwargs):
        '''
        Make a map of the grid using the `show` method and save the image to file without
        labels, title and colorbar.
        The parameters are the same as `show`, expect there is no colorbar and
        there are these additional parameters:
        output_file : string
            Path to the image file. The extension controls the output format.
        scale : float
            Coefficient to set the size of the output image. With a scale of 1,
            the image will have the same size (columns and rows) as the grid.
            To make an image smaller than the grid, use a scale smaller than 1.
        '''
        if 'origin' in kwargs:
            ax = graphics.imshow_hs(self, ax=None, cmap=cmap, cmap_norm=cmap_norm,
                                    hs=hs, zf=zf, azdeg=azdeg, altdeg=altdeg,
                                    dx=dx, dy=dy, hs_contrast=hs_contrast,
                                    cmap_brightness=cmap_brightness, blend_mode=blend_mode,
                                    alpha=alpha, contours=contours,
                                    colorbar=False, **kwargs)
        else:
            # set origin to ensure that both grid and contours get the same origin
            ax = graphics.imshow_hs(self, ax=None, cmap=cmap, cmap_norm=cmap_norm,
                                    hs=hs, zf=zf, azdeg=azdeg, altdeg=altdeg,
                                    dx=dx, dy=dy, hs_contrast=hs_contrast,
                                    cmap_brightness=cmap_brightness, blend_mode=blend_mode,
                                    alpha=alpha, contours=contours,
                                    colorbar=False, origin='upper', **kwargs)

        fig1 = ax.get_figure()
        graphics.save_image(output_file,
                            fig=fig1,
                            size=(scale*self.ncols, scale*self.nrows))

        # clear figure to avoid displaying the result
        fig1.clear()

        print('The grid was successfully saved as an image in {}'.format(output_file))


    ### Filters
    def smooth(self, method='SG', deg=3, win=5, doEdges=True, sigma=1):
        '''Smoothing filters.
        '''
        if method.upper() == 'SG':
            output = transforms.savgol_smooth(self.data, deg=deg,
                                              win=win, doEdges=doEdges)

        elif method.lower() == 'gaussian':
            output = transforms.gauss(self.data, sigma=sigma)

        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        return Grid(output, self.transform, name=self.name+'_smooth')


    def laplacian(self):
        '''Calculate the Laplacian using 2D convolution.
        '''
        output = transforms.laplacian(self.data, self.cellsize)
        return Grid(output, self.transform, name=self.name+'_laplacian')


    ### Derivatives

    # horizontal derivatives
    def dx(self, method='SG', deg=3, win=5, doEdges=True, fs_tap=5, **kwargs):
        '''Calculate first horizontal derivative with various methods.
        '''
        if method.upper() == 'SG':
            output = transforms.savgol_deriv(self.data, self.cellsize,
                                             direction='dx', deg=deg,
                                             win=win, doEdges=doEdges)

        elif method.upper() == 'FS':
            output = transforms.fs_deriv(self.data, self.cellsize,
                                         direction='dx', tap=fs_tap)

        elif method.lower() == 'fourier':
            output = transforms.fourier_transform(self.data, self.cellsize,
                                                  trans='dx', **kwargs)
        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        return Grid(output, self.transform, name=self.name+'_dx')


    def dx2(self, method='SG', deg=4, win=5, doEdges=True, fs_tap=5, **kwargs):
        '''Calculate second horizontal derivative with various methods.
        '''
        if method.upper() == 'SG':
            output = transforms.savgol_deriv(self.data, self.cellsize,
                                             direction='dx2', deg=deg,
                                             win=win, doEdges=doEdges)

        elif method.upper() == 'FS':
            output = transforms.fs_deriv(self.data, self.cellsize,
                                         direction='dx2', tap=fs_tap)

        elif method.lower() == 'fourier':
            output = transforms.fourier_transform(self.data, self.cellsize,
                                                  trans='dx', order=2, **kwargs)
        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        return Grid(output, self.transform, name=self.name+'_dx2')


    def dy(self, method='SG', deg=3, win=5, doEdges=True, fs_tap=5, **kwargs):
        '''Calculate first horizontal derivative with various methods.
        '''
        if method.upper() == 'SG':
            output = transforms.savgol_deriv(self.data, self.cellsize,
                                             direction='dy', deg=deg,
                                             win=win, doEdges=doEdges)

        elif method.upper() == 'FS':
            output = transforms.fs_deriv(self.data, self.cellsize,
                                         direction='dy', tap=fs_tap)

        elif method.lower() == 'fourier':
            output = transforms.fourier_transform(self.data, self.cellsize,
                                                  trans='dy', **kwargs)
        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        return Grid(output, self.transform, name=self.name+'_dy')


    def dy2(self, method='SG', deg=4, win=5, doEdges=True, fs_tap=5, **kwargs):
        '''Calculate second horizontal derivative with various methods.
        '''
        if method.upper() == 'SG':
            output = transforms.savgol_deriv(self.data, self.cellsize,
                                             direction='dy2', deg=deg,
                                             win=win, doEdges=doEdges)

        elif method.upper() == 'FS':
            output = transforms.fs_deriv(self.data, self.cellsize,
                                         direction='dy2', tap=fs_tap)

        elif method.lower() == 'fourier':
            output = transforms.fourier_transform(self.data, self.cellsize,
                                                  trans='dy', order=2, **kwargs)
        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        return Grid(output, self.transform, name=self.name+'_dy2')


    def dxdy(self, method='SG', deg=3, win=5, doEdges=True, fs_tap=5, **kwargs):
        '''Calculate second horizontal derivative with various methods.
        '''
        if method.upper() == 'SG':
            output = transforms.savgol_deriv(self.data, self.cellsize,
                                             direction='dxdy', deg=deg,
                                             win=win, doEdges=doEdges)

        elif method.upper() == 'FS':
            output = transforms.fs_deriv(self.data, self.cellsize,
                                         direction='dxdy', tap=fs_tap)

        elif method.lower() == 'fourier':
            output = transforms.fourier_transform(self.data, self.cellsize,
                                                  trans='dxdy', **kwargs)
        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        return Grid(output, self.transform, name=self.name+'_dxdy')


    # vertical derivative
    def dz(self, method='fourier', order=1, **kwargs):
        '''Calculate the vertical derivative with various methods.
        '''
        if method.lower() == 'fourier':
            output = transforms.fourier_transform(self.data, self.cellsize,
                                                  trans='dz', order=order, **kwargs)
        elif method.lower() == 'isvd':
            dz_grid = self.isvd(method='SG', order=order, **kwargs)
            output = dz_grid.data
        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        return Grid(output, self.transform, name='{}_dz{}'.format(self.name, order))


    # vertical derivative with ISVD method (makes use of the Laplace eq.)
    def isvd(self, method='SG', order=1, deg=3, win=5, doEdges=True,
             fs_tap=5, **kwargs):
        '''Calculate the vertical derivative with the ISVD method.
        '''
        if method.lower() in ['sg', 'fs', 'fourier']:
            output = transforms.isvd(self.data, self.cellsize, method=method,
                                     order=order, deg=deg, win=win, doEdges=doEdges,
                                     fs_tap=fs_tap, **kwargs)

        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        return Grid(output, self.transform, name='{}_dz{}'.format(self.name, order))


    # vertical integral
    def vi(self, method='fourier', order=1, eps=1e-6, **kwargs):
        '''Calculate the vertical integral.
        '''
        if method.lower() == 'fourier':
            output = transforms.fourier_transform(self.data, self.cellsize,
                                                  trans='vi', order=order, eps=eps, **kwargs)
        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        return Grid(output, self.transform, name='{}_vi{}'.format(self.name, order))

    ### Transforms

    # horizontal gradient magnitude
    def hgm(self, method='SG', deg=3, win=5, doEdges=True, fs_tap=5, **kwargs):
        '''Calculate the horizontal gradient magnitude with various methods.
        '''
        if method.upper() == 'SG':
            dx1 = transforms.savgol_deriv(self.data, self.cellsize,
                                          direction='dx', deg=deg,
                                          win=win, doEdges=doEdges)
            dy1 = transforms.savgol_deriv(self.data, self.cellsize,
                                          direction='dy', deg=deg,
                                          win=win, doEdges=doEdges)

        elif method.upper() == 'FS':
            dx1 = transforms.fs_deriv(self.data, self.cellsize,
                                      direction='dx', tap=fs_tap)
            dy1 = transforms.fs_deriv(self.data, self.cellsize,
                                      direction='dy', tap=fs_tap)

        elif method.lower() == 'fourier':
            dx1 = transforms.fourier_transform(self.data, self.cellsize,
                                               trans='dx', order=1, **kwargs)
            dy1 = transforms.fourier_transform(self.data, self.cellsize,
                                               trans='dy', order=1, **kwargs)
        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        return Grid(np.sqrt(dx1*dx1 + dy1*dy1),
                    self.transform, name=self.name+'_hgm')


    # tilt angle
    def tilt(self, hgm_method='SG', dz_method='isvd', deg=3, win=5,
             doEdges=True, fs_tap=5, alpha=1):
        '''Calculate the tilt angle of anomalies.
        The alpha option implements the downward continuation of the tilt angle
        as described by Cooper (2016).

        Reference
        ---------
        Cooper, G., 2016. The downward continuation of the tilt angle. Near
        Surf. Geophys. 14, 385–390. doi:10.3997/1873-0604.2016022
        '''

        # horizontal gradient
        hgm_grid = self.hgm(method=hgm_method, deg=deg, win=win, doEdges=doEdges,
                            fs_tap=fs_tap)
        # vertical derivative
        if dz_method.lower() == 'isvd':
            dz_grid = self.isvd(method=hgm_method, order=1, deg=deg, win=win,
                                doEdges=doEdges, fs_tap=fs_tap)
        else:
            dz_grid = self.dz(method='fourier', order=1)
        # calculate tilt angle (in degrees)
        output = np.arctan(alpha * dz_grid.data / hgm_grid.data) * 180 / np.pi

        return Grid(output, self.transform, name=self.name+'_tilt')


    # total gradient
    def tg(self, method='SG', dz_method='isvd', deg=3, win=5, doEdges=True,
           fs_tap=5, **kwargs):
        '''Calculate the total gradient with various methods.
        '''
        # vertical derivative
        if dz_method.lower() == 'isvd':
            dz_grid = self.isvd(method=method, order=1, deg=deg, win=win,
                                doEdges=doEdges, fs_tap=fs_tap)

        else:
            dz_grid = self.dz(method='fourier', order=1)

        # horizontal derivatives
        if method.upper() == 'SG':
            dx1 = transforms.savgol_deriv(self.data, self.cellsize,
                                          direction='dx', deg=deg,
                                          win=win, doEdges=doEdges)
            dy1 = transforms.savgol_deriv(self.data, self.cellsize,
                                          direction='dy', deg=deg,
                                          win=win, doEdges=doEdges)

        elif method.upper() == 'FS':
            dx1 = transforms.fs_deriv(self.data, self.cellsize,
                                      direction='dx', tap=fs_tap)
            dy1 = transforms.fs_deriv(self.data, self.cellsize,
                                      direction='dy', tap=fs_tap)

        elif method.lower() == 'fourier':
            dx1 = transforms.fourier_transform(self.data, self.cellsize,
                                               trans='dx', order=1, **kwargs)
            dy1 = transforms.fourier_transform(self.data, self.cellsize,
                                               trans='dy', order=1, **kwargs)

        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        # calculate total gradient
        output = np.sqrt(dx1*dx1 + dy1*dy1 + dz_grid.data*dz_grid.data)

        return Grid(output, self.transform, name=self.name+'_TG')


    # local wavenumber
    def lw(self, method='SG', dz_method='isvd', deg=4, win=5, doEdges=True,
           fs_tap=5, **kwargs):
        '''Calculate the local wavenumber with various methods.
        '''
        # vertical derivative
        if dz_method.lower() == 'isvd':
            dz_grid = self.isvd(method=method, order=1, deg=deg, win=win,
                                doEdges=doEdges, fs_tap=fs_tap)
            dzdz_grid = self.isvd(method=method, order=2, deg=deg, win=win,
                                  doEdges=doEdges, fs_tap=fs_tap)
        else:
            dz_grid = self.dz(method='fourier', order=1)
            dzdz_grid = self.dz(method='fourier', order=2)

        # horizontal derivatives
        if method.upper() == 'SG':
            dx1 = transforms.savgol_deriv(self.data, self.cellsize,
                                          direction='dx', deg=deg,
                                          win=win, doEdges=doEdges)
            dy1 = transforms.savgol_deriv(self.data, self.cellsize,
                                          direction='dy', deg=deg,
                                          win=win, doEdges=doEdges)
            dxdz1 = transforms.savgol_deriv(dz_grid.data, self.cellsize,
                                            direction='dx', deg=deg,
                                            win=win, doEdges=doEdges)
            dydz1 = transforms.savgol_deriv(dz_grid.data, self.cellsize,
                                            direction='dy', deg=deg,
                                            win=win, doEdges=doEdges)
        elif method.upper() == 'FS':
            dx1 = transforms.fs_deriv(self.data, self.cellsize,
                                      direction='dx', tap=fs_tap)
            dy1 = transforms.fs_deriv(self.data, self.cellsize,
                                      direction='dy', tap=fs_tap)
            dxdz1 = transforms.fs_deriv(dz_grid.data, self.cellsize,
                                        direction='dx', tap=fs_tap)
            dydz1 = transforms.fs_deriv(dz_grid.data, self.cellsize,
                                        direction='dy', tap=fs_tap)

        elif method.lower() == 'fourier':
            dx1 = transforms.fourier_transform(self.data, self.cellsize,
                                               trans='dx', order=1, **kwargs)
            dy1 = transforms.fourier_transform(self.data, self.cellsize,
                                               trans='dy', order=1, **kwargs)
            dxdz1 = transforms.fourier_transform(dz_grid.data, self.cellsize,
                                                 trans='dx', order=1, **kwargs)
            dydz1 = transforms.fourier_transform(dz_grid.data, self.cellsize,
                                                 trans='dy', order=1, **kwargs)

        else:
            raise ValueError('Method {} has not been recognised.'.format(method))

        # calculate local wavenumber
        output = ((dxdz1*dx1 + dydz1*dy1 + dzdz_grid.data*dz_grid.data)
                  /(dx1*dx1 + dy1*dy1 + dz_grid.data*dz_grid.data))

        return Grid(output, self.transform, name=self.name+'_LW')


    # TAHG (Tilt Angle of the Horizontal Gradient)
    def tahg(self, method='SG', dz_method='isvd', deg=4, win=5, doEdges=True,
             fs_tap=5, alpha=1, **kwargs):
        '''Calculate the tahg transform with various methods.

        Reference
        ---------
        Ferreira, F.J.F., de Souza, J., de B. e S. Bongiolo, A., de Castro,
        L.G., 2013. Enhancement of the total horizontal gradient of magnetic
        anomalies using the tilt angle. Geophysics 78, J33–J41.
        doi:10.1190/geo2011-0441.1
        '''
        # Calculate the HGM
        grid_hgm = self.hgm(method=method, deg=deg, win=win, doEdges=doEdges,
                            fs_tap=fs_tap, **kwargs)

        # Calculate the tilt angle of the HGM
        grid_tahg = grid_hgm.tilt(hgm_method=method, dz_method=dz_method,
                                  deg=deg, win=win, doEdges=doEdges,
                                  fs_tap=fs_tap, alpha=alpha, **kwargs)
        grid_tahg.name = self.name+'_TAHG'

        return grid_tahg


    ## Upward continuation
    def up(self, z=500, **kwargs):
        '''Upward continuation. The calculation is done in the frequency domain.

        Parameters
        ----------
        z: float
            Amount of upward continuation. In practice, this is the value
            to add to the observation height (same units as the cell size).
        **kwargs are passed to the Fourier transform.
        '''
        output = transforms.fourier_transform(self.data, self.cellsize,
                                              trans='upcont', z=z, **kwargs)

        return Grid(output, self.transform, name=self.name+'_UC{}'.format(z))


    ## High-pass filter by upward continuation
    def hp_filter_uc(self, z=5000, **kwargs):
        '''Apply a high-pass filter by subtracting an upward continued version
        of the data.

        Parameters
        ----------
        z: float
            Amount of upward continuation. The larger the value, the larger the
            part of the signal that is passed through the filter. Use a small value
            to increase the effect of the filter.
        **kwargs are passed to the Fourier transform.
        '''
        upCont1 = transforms.fourier_transform(self.data, self.cellsize,
                                               trans='upcont', z=z, **kwargs)
        output = self.data - upCont1

        return Grid(output, self.transform, name=self.name+'_HPUC{}'.format(z))

#==============================================================================
# Functions
#==============================================================================
def from_dataset(dataset, band=1, crs=None, name=None,
                 nodata_value=None, scale_factor=None):
    '''
    Create a new grid object from a rasterio dataset.
    Used by interpies.open().
    '''
    # extract name of the dataset (without extension)
    if name is None:
        name = os.path.basename(os.path.splitext(dataset.name)[0])
    # nodata value
    if nodata_value is None:
        nodata_value = dataset.nodata

    # coordinate system
    if crs is None and dataset.crs is not None:
        if dataset.crs.is_epsg_code:
            # get the EPSG code
            crs = dataset.crs['init']
        else:
            # get a PROJ.4 string
            crs = dataset.crs.to_string()

    # read the data from the specified band and apply scaling
    if scale_factor is not None:
        data = scale_factor * dataset.read(band)
    else:
        data = dataset.read(band)

    return Grid(data, dataset.transform, name=name,
                nodata_value=nodata_value, filename=dataset.name, crs=crs)

def from_array(array, west=0, north=0, cellsize=100, y_cellsize=100, crs=None,
               name='Unknown', filename='Unknown', nodata_value=None):
    '''
    Create a new grid object from a numpy array.
    Used by interpies.open().
    '''
    # name
    if name == 'Unknown':
        if filename != 'Unknown':
            name = os.path.basename(os.path.splitext(filename)[0])
        else:
            name = 'Unknown'
            filename = 'Unknown'

    # nodata value
    if nodata_value is None:
        nodata_value = np.nan

    # transform
    transf = rasterio.transform.from_origin(west, north, cellsize, y_cellsize)

    return Grid(array, transf, name=name,
                nodata_value=nodata_value, filename=filename, crs=crs)
