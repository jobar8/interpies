"""Tests for the Grid class."""

import numpy as np
import pytest

from interpies.grid import Grid


@pytest.fixture
def grid1() -> Grid:
    data = np.linspace([0, 0], [49, 49]).reshape((10, 10))
    return Grid(data=data)


def test_grid_instance():
    data = np.linspace([0, 0], [49, 49]).reshape((10, 10))
    grid = Grid(data=data)
    assert isinstance(grid, Grid)
    assert grid.crs == 'Unknown'
    assert grid.ncols == 10
    assert grid.nrows == 10
    assert grid.name == 'Unknown'
    assert grid.filename == 'Unknown'
    assert grid.cellsize == 100
    assert grid.y_cellsize == 100
    assert grid.xll == 50.0
    assert grid.yll == -950.0
    assert grid.nodata is None
    assert grid.extent == [0.0, 1000.0, -1000.0, 0.0]


def test_clip1(grid1):
    """Test clipping with new area smaller than original."""
    new_grid = grid1.clip(110, 800, -805, -500)
    assert isinstance(new_grid, Grid)
    assert new_grid.name == 'Unknown_clip'
    assert new_grid.extent == [100.0, 900.0, -900.0, -500.0]
    assert new_grid.xll == 150.0
    assert new_grid.yll == -850.0
    assert new_grid.ncols == 8
    assert new_grid.nrows == 4


def test_clip2(grid1: Grid):
    """Test clipping with new area larger than original."""
    new_grid = grid1.clip(-330, 1323.5, -1500, 100)
    assert new_grid.extent == [0.0, 1000.0, -1000.0, 0.0]
    assert new_grid.xll == 50.0
    assert new_grid.yll == -950.0
    assert new_grid.ncols == 10
    assert new_grid.nrows == 10


def test_clip3(grid1: Grid):
    """Test clipping with new area outside original."""
    new_grid = grid1.clip(1500, 2000.5, 0, 1000)
    assert new_grid.extent == [1500.0, 1500.0, -100.0, 0.0]
    assert new_grid.xll == 1550.0
    assert new_grid.yll == -50.0
    assert new_grid.ncols == 0
    assert new_grid.nrows == 1
    assert new_grid.data.size == 0
