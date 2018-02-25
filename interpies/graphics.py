# -*- coding: utf-8 -*-
"""
Interpies - a libray for the interpretation of gravity and magnetic data.

graphics.py:
    Functions for creating and manipulating graphics, colormaps and plots.

@author: Joseph Barraud
Geophysics Labs, 2017
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from skimage import exposure

# import local modules
import interpies
import interpies.colors as icolors

# temporary solution to silence a warning issued by Numpy when called by matplotlib imshow function
warnings.filterwarnings('ignore', category=FutureWarning)

#==============================================================================
# Functions for loading and modifying colormaps
#==============================================================================
def make_colormap(table, name='CustomMap'):
    """
    Return a LinearSegmentedColormap. The colormap is also registered with
    plt.register_cmap(cmap=my_cmap)

    Parameters
    ----------
    table : a sequence of RGB tuples.
        Values need to be either floats between 0 and 1, or
        integers between 0 and 255.
    """
    if np.any(table > 1):
        table = table / 255.
    cdict = {'red': [], 'green': [], 'blue': []}
    N = float(len(table))-1
    for i, rgb in enumerate(table):
        red, gre, blu = rgb
        cdict['red'].append([i/N, red, red])
        cdict['green'].append([i/N, gre, gre])
        cdict['blue'].append([i/N, blu, blu])

    new_cmap = mcolors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=new_cmap)
    return new_cmap


def cmap_to_array(cmap, n=256):
    """
    Return a nx3 array of RGB values that defines a colormap or generate it from
    a colormap object.
    Input is colormap name (if recognised) or matplotlib cmap object.
    """
    # first assume cmap is a string
    if cmap in icolors.datad: # additional colormaps in interpies.colors module
        cm_array = np.asarray(icolors.datad[cmap])
    elif cmap in cm.cmap_d: # matplotlib colormaps + the new ones (viridis, inferno, etc.)
        cmap = cm.cmap_d[cmap]
        cm_array = cmap(np.linspace(0, 1, n))[:, :3]
    # now assume cmap is a colormap object
    else:
        try:
            cm_array = cmap(np.linspace(0, 1, n))[:, :3] # remove alpha column
        except:
            raise ValueError('Colormap {} has not been recognised'.format(cmap))

    return cm_array


def load_cmap(cmap='geosoft'):
    """
    Return a colormap object.
    If input is a string, load first the colormap, otherwise return the cmap unchanged.
    """
    # first suppose input is the name of the colormap
    if cmap in icolors.datad: # one of the additional colormaps in interpies colors module
        cm_list = icolors.datad[cmap]
        new_cm = mcolors.LinearSegmentedColormap.from_list(cmap, cm_list)
        plt.register_cmap(cmap=new_cm)
        return new_cm
    elif cmap in cm.cmap_d: # matplotlib colormaps + the new ones (viridis, inferno, etc.)
        return cm.get_cmap(cmap)
    elif isinstance(cmap, mcolors.Colormap):
        return cmap
    else:
        raise ValueError('Colormap {} has not been recognised'.format(cmap))


def plot_cmap(name='geosoft', n=256):
    '''
    Make a checkerboard plot of the colours in a palette.

    Parameters
    ----------
    name : str
        Name of the colormap to plot.
    n : int, optional
        Number of cells to use. Note that the closest power of 2 is actually used
        as the plot is a square.
    '''
    ncols = int(np.sqrt(n))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.arange(ncols**2).reshape(ncols, ncols),
              cmap=load_cmap(name),
              interpolation="nearest", aspect="equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)


def equalize_colormap(cmap, data, name='EqualizedMap'):
    '''
    Re-map a colormap according to a cumulative distribution. This is used to
    perform histogram equalization of an image by changing the colormap
    instead of the image. *This is not strickly speaking the equalization of the
    colormap itself*.
    The cdf and bins is calculated from the input image, as if carrying out
    the histogram equalization of that image. In effect, the cdf becomes integrated
    to the colormap as a mapping function by redistributing the indices of the
    input colormap.

    Parameters
    ----------
    cmap : string or colormap object
        Input colormap to remap.
    data : array
        Input data
    '''
    # first retrieve the color table (lists of RGB values) behind the input colormap
    cm_array = cmap_to_array(cmap, n=256)

    # perform histogram equalization of the data using scikit-image function.
    # bins : centers of bins, cdf : values of cumulative distribution function.
    cdf, bins = exposure.cumulative_distribution(
                         data[~np.isnan(data)].flatten(), nbins=256)

    # normalize the bins to interval (0,1)
    bins_norm = (bins - bins.min())/np.float(bins.max() - bins.min())

    # calculate new indices by applying the cdf as a function on the old indices
    # which are initially regularly spaced.
    old_indices = np.linspace(0, 1, len(cm_array))
    new_indices = np.interp(old_indices, cdf, bins_norm)

    # make sure indices start with 0 and end with 1
    new_indices[0] = 0.0
    new_indices[-1] = 1.0

    # remap the color table
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, n in enumerate(new_indices):
        red, gre, blu = cm_array[i]
        cdict['red'].append([n, red, red])
        cdict['green'].append([n, gre, gre])
        cdict['blue'].append([n, blu, blu])

    # return new colormap
    return mcolors.LinearSegmentedColormap(name, cdict)


def clip_colormap(cm_array, data, min_percent=2, max_percent=98, name='ClippedMap'):
    '''
    Modify the colormap so that the image of the data looks clipped at extreme
    values.
    Clipping boundaries are specified by percentiles and calculated from the
    input data. These boundaries are then "transfered" to the colormap.
    '''
    # remove NaNs
    valid_data = data[~np.isnan(data)]

    # calculate boundaries from data
    data_min, data_max = np.percentile(valid_data, (min_percent, max_percent))

    # calculate corresponding values on a scale from 0 to 1
    imin = (data_min - valid_data.min()) / (valid_data.max() - valid_data.min())
    imax = (data_max - valid_data.min()) / (valid_data.max() - valid_data.min())

    # calculate the number of indices to add to accommodate clipped values
    n_new = len(cm_array) / (imax - imin)
    n_left = int(imin * n_new)
    n_right = int(n_new - imax * n_new)

    # calculate new indices
    new_indices = np.linspace(0, 1, len(cm_array) + n_left + n_right)

    # remap the color table
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, n in enumerate(new_indices):
        if i < n_left:
            red, gre, blu = cm_array[0]
            cdict['red'].append([n, red, red])
            cdict['green'].append([n, gre, gre])
            cdict['blue'].append([n, blu, blu])
        elif i >= len(cm_array) + n_left:
            red, gre, blu = cm_array[-1]
            cdict['red'].append([n, red, red])
            cdict['green'].append([n, gre, gre])
            cdict['blue'].append([n, blu, blu])
        else:
            red, gre, blu = cm_array[i - n_left]
            cdict['red'].append([n, red, red])
            cdict['green'].append([n, gre, gre])
            cdict['blue'].append([n, blu, blu])

    # return new colormap
    return mcolors.LinearSegmentedColormap(name, cdict)


def modify_colormap(cmap, data=None, modif='autolevels',
                    min_percent=2, max_percent=98, brightness=1.0):
    '''
    Modify a colormap by clipping or rescaling, according to statistics of the
    input data or to fixed parameters. Also implement brightness control.
    '''
    # get the name of the colormap if input is colormap object
    if isinstance(cmap, mcolors.Colormap):
        cm_name = cmap.name
    else:
        cm_name = cmap

    # retrieve the color table (lists of RGB values) behind the input colormap
    cm_array = cmap_to_array(cmap, n=256)

    # modify color table
    if modif == 'autolevels':
        return clip_colormap(cm_array,
                             data,
                             min_percent=min_percent,
                             max_percent=max_percent)
    elif modif == 'brightness':
        # convert brightness to gamma value
        gamma = np.exp(1/brightness - 1)
        normed_cm_array = exposure.adjust_gamma(cm_array, gamma=gamma)
    else:
        normed_cm_array = cm_array

    # create new colormap
    new_cm = mcolors.LinearSegmentedColormap.from_list(cm_name + '_n', normed_cm_array)

    return new_cm

#===============================================================================
# Functions for displaying grid data
#===============================================================================
def stats_boundaries(data, std_range=1, step=1):
    '''
    Return a list of statistical quantities ordered in increasing order: min, mean, max
    and the standard deviation intervals in between.
    These are intended to be used for axis ticks in plots.

    Parameters
    ----------
    data : array-like
        Input data.
    std_range : int, optional
        Extent of the range from the mean as a multiple of the standard deviation.
    step : float, optional
        Size of the interval, as a fraction of the standard deviation. Must be <= `nSigma`.
    '''
    mean = np.nanmean(data)
    sigma = np.nanstd(data)
    new_ticks = mean + sigma*np.arange(-std_range, std_range+step, step)
    # make sure the boundaries don't go over min and max
    new_ticks = np.unique(new_ticks.clip(np.nanmin(data), np.nanmax(data)))

    return [np.nanmin(data)] + list(new_ticks) + [np.nanmax(data)]


def alpha_blend(rgb, intensity, alpha=0.7):
    """
    Combines an RGB image with an intensity map using "alpha" transparent blending.
    https://en.wikipedia.org/wiki/Alpha_compositing

    Parameters
    ----------
    rgb : ndarray
        An MxNx3 RGB array of floats ranging from 0 to 1 (color image).
    intensity : ndarray
        An MxNx1 array of floats ranging from 0 to 1 (grayscale image).
    alpha : float
        This controls the transparency of the rgb image. 1.0 is fully opaque
        while 0.0 is fully transparent.

    Returns
    -------
    rgb : ndarray
        An MxNx3 RGB array representing the combined images.
    """

    return alpha*rgb + (1 - alpha)*intensity


def imshow_hs(source, ax=None, cmap='geosoft', cmap_norm='equalize', hs=True,
              zf=10, azdeg=45, altdeg=45, dx=1, dy=1, hs_contrast=1.5, cmap_brightness=1.0,
              blend_mode='alpha', alpha=0.7, contours=False, colorbar=True,
              cb_contours=False, cb_ticks='linear', std_range=1, figsize=(8, 8),
              title=None, **kwargs):
    '''
    Display an array or a grid with optional hillshading and contours.
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
        Add lines corresponding to contours values on the colorbar.
    figsize: tuple
        Dimensions of the figure: width, height in inches.
        If not provided, the default is (8, 8).
    title: string
        String to display as a title above the plot. If the source is a grid
        object, the title is taken by default from the name of the grid.
    kwargs : other optional arguments
        Can be used to pass other arguments to `plt.imshow()`, such as 'origin'
        and 'extent', or to the colorbar('shrink'), the title
        ('fontweight' and 'fontsize'), or the contours ('colors').

    Returns
    -------
    ax : Matplotlib Axes instance.

    Notes
    -----
    This function exploits the hillshading capabilities implemented in
    matplotlib.colors.LightSource. A new blending mode is added (alpha compositing,
    see https://en.wikipedia.org/wiki/Alpha_compositing).

    '''
    # get extra information if input data is grid object (grid.grid)
    # `extent` is added to the kwargs of the imshow function
    if isinstance(source, interpies.Grid):
        kwargs['extent'] = source.extent
        data = source.data
        if title is None:
            if source.name != 'Unknown':
                title = source.name
    else:
        data = source.copy()

    ## Extract keywords - using pop() also removes the key from the dictionary
    # keyword for the colorbar
    cb_kwargs = dict(shrink=kwargs.pop('shrink', 0.6))

    # keywords for the title
    title_kwargs = dict(fontweight=kwargs.pop('fontweight', None),
                        fontsize=kwargs.pop('fontsize', 'large'))

    # keyword arguments that can be passed to ls.shade
    shade_kwargs = dict(norm=kwargs.get('norm'),
                        vmin=kwargs.get('vmin'),
                        vmax=kwargs.get('vmax'))

    # keywords for cmap normalisation
    min_percent = kwargs.pop('min_percent', 2)
    max_percent = kwargs.pop('max_percent', 98)

    # keywords for contours
    ct_colors = kwargs.pop('ct_colors', 'k')
    ct_cmap = kwargs.pop('ct_cmap', None)

    # modify colormap if required
    if cmap_norm in ['equalize', 'equalise', 'equalization', 'equalisation']:
        # equalisation
        my_cmap = equalize_colormap(cmap, data)

    elif cmap_norm in ['auto', 'autolevels']:
        # clip colormap
        my_cmap = modify_colormap(cmap, data, modif='autolevels',
                                  min_percent=min_percent, max_percent=max_percent)
    else:
        # colormap is loaded unchanged from the input name
        my_cmap = load_cmap(cmap)  # raise error if name is not recognised

    # apply brightness control
    if cmap_brightness != 1.0:
        my_cmap = modify_colormap(my_cmap, modif='brightness', brightness=cmap_brightness)

    # create figure or retrieve the one already defined
    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # convert input data to a masked array
    data = np.ma.masked_array(data, np.isnan(data))

    # add array to figure with hillshade or not
    if hs:
        # flip azimuth upside down if grid is also flipped
        if 'origin' in kwargs:
            if kwargs['origin'] == 'lower':
                azdeg = 180 - azdeg

        # create light source
        ls = mcolors.LightSource(azdeg, altdeg)

        # calculate hillshade and combine the colormapped data with the intensity
        if alpha == 0:
            # special case when only the shaded relief is needed without blending
            rgb = ls.hillshade(data, vert_exag=zf, dx=dx, dy=dy, fraction=hs_contrast)
            kwargs['cmap'] = 'gray'

        elif blend_mode == 'alpha':
            # transparency blending
            rgb = ls.shade(data, cmap=my_cmap, blend_mode=alpha_blend,
                           vert_exag=zf, dx=dx, dy=dy,
                           fraction=hs_contrast, alpha=alpha, **shade_kwargs)

        else:
            # other blending modes from matplotlib function
            rgb = ls.shade(data, cmap=my_cmap, blend_mode=blend_mode,
                           vert_exag=zf, dx=dx, dy=dy,
                           fraction=hs_contrast, **shade_kwargs)

        # finally plot the array
        ax.imshow(rgb, **kwargs)

    else:
        # display data without hillshading
        im = ax.imshow(data, cmap=my_cmap, **kwargs)

    # add contours
    levels = None
    if isinstance(contours, bool):
        if contours:
            levels = 32
    else:
        levels = contours
        contours = True
    if levels is not None:
        # remove cmap keyword that might have been added earlier
        _ = kwargs.pop('cmap', None)
        conts = plt.contour(data, levels, linewidths=0.5,
                            colors=ct_colors, linestyles='solid',
                            cmap=ct_cmap, **kwargs)

    # add colorbar
    if colorbar and alpha != 0:
        if hs:
            # Use a proxy artist for the colorbar
            im = ax.imshow(data, cmap=my_cmap, **kwargs)
            im.remove()
        # draw colorbar
        if cb_ticks == 'linear': # normal equidistant ticks on a linear scale
            cb1 = plt.colorbar(im, ax=ax, **cb_kwargs)
        else: # show ticks at min, max, mean and standard deviation interval
            new_ticks = stats_boundaries(data, std_range, std_range)
            cb1 = plt.colorbar(im, ax=ax, ticks=new_ticks, **cb_kwargs)

        # add optional contour lines on colorbar
        if contours and cb_contours:
            cb1.add_lines(conts)

        cb1.update_normal(im)

    # add title
    if title:
        ax.set_title(title, **title_kwargs)

    # return Axes instance for re-use
    return ax


def save_image(output_file, fig=None, size=None, dpi=100):
    '''
    Save a Matplotlib figure as an image without borders or frames. The format
    is controlled by the extension of the output file name.

    Parameters
    ----------
    output_file : string
        Path to output file.
    fig : Matplotlib figure instance
        Figure you want to save as the image
    size : tuple (w, h)
        Width and height of the output image in pixels.
    dpi : integer
        Image resolution.
    '''
    if fig is None:
        fig = plt.gcf()
    ax = fig.gca()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])

    if size:
        w, h = size
        fig.set_size_inches(w/dpi, h/dpi, forward=False)

    fig.savefig(output_file, dpi=dpi)
