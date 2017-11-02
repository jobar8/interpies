# Interpies

`interpies` is a collection of functions to read and analyse geophysical data, especially non-seismic data such as magnetic and gravity data.

## Table of Contents

* [Getting Started](##GettingStarted)
    * [Requirements](###Prerequisits)
    * [Installation](###Installation)
* [Examples](##Examples)
* [Documentation](##Documentation)


## Getting Started

### Requirements

Interpies requires `Python 3.x` and makes use of the following libraries:

* `numpy`
* `matplotlib`
* `rasterio` version > 1.0
* `gdal`
* `scikit-learn`
* `scikit-image`

Optional:

* `obspy` for reading and writing SEG-Y files (seismic data)
* `geopandas` for reading survey line data
* `ipykernel` for working with `interpies` in Jupyter notebooks


### Installation

#### Dependencies

I recommend using [Anaconda](https://www.continuum.io/what-is-anaconda) for the installation of both Python and most of the dependencies.

Once Anaconda has been installed, I would suggest creating a new environment for working with `interpies`. You could start with this command:

`conda create --name interpies gdal scikit-learn scikit-image matplotlib ipykernel obspy python=3.6`

Next, install `rasterio`. You could try using `conda install rasterio`. However, the only version available on conda-forge might be the old 0.36. The alpha version 1.09a or better is required for `interpies` to work. So carefully check which version is going to be installed first. 

On **Windows**, if the version does not match, simply download the binaries for the required version from Christoph Gohlke's [website](http://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio). Then run, for example:

`pip install rasterio-1.0a9-cp36-cp36m-win_amd64.whl`

And that should do. If you encounter other problems with this part of the installation, please refer to the [rasterio installation](https://mapbox.github.io/rasterio/installation.html).

Optionally, you could also install `geopandas`, which is great for reading line data from geophysical surveys. And don't forget to install `ipykernel` to run the notebooks in the `interpies` environment.


#### interpies

Installing `interpies` itself is done directly with:

`pip install interpies`

Or you could do it manually by first cloning the current repository:

`$ git clone https://github.com/jobar8/interpies.git`

Then run the following command in the repository directory:

`$ python setup.py install`

## Examples

The basic usage of `interpies` is to load gridded data into a *grid* object, which then gives access to various methods for transforming and displaying the data. So, loading magnetic data and creating a map with the grid is simply done with:

```
import interpies 
grid1 = interpies.open(r'..\data\brtpgrd.gxf')
grid1.show()
```

![image of magnetic data](/doc/mag_survey_example.png)


For more advanced examples, please see the notebooks.


## Documentation

Under construction.
