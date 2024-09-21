"""Tests for the graphics module."""

import pytest
import matplotlib
import matplotlib.colors as mcolors

import numpy as np
from interpies.graphics import cmap_to_array, load_cmap


def test_cmap_to_array():
    cmap = matplotlib.colormaps['viridis']
    array1 = cmap_to_array('viridis')
    array2 = cmap_to_array(cmap)

    assert isinstance(array1, np.ndarray)
    assert isinstance(array2, np.ndarray)
    assert array1.shape == (256, 3)
    np.testing.assert_equal(array1[0], [0.267004, 0.004874, 0.329415])
    np.testing.assert_equal(array1, array2)

    with pytest.raises(KeyError):
        array1 = cmap_to_array('not a colormap')
    with pytest.raises(ValueError):
        array1 = cmap_to_array([1, 2, 3, 4])  # type: ignore


def test_load_cmap():
    cmap1 = load_cmap('viridis')
    assert isinstance(cmap1, mcolors.Colormap)
    cmap2 = load_cmap('geosoft')
    assert isinstance(cmap2, mcolors.Colormap)
