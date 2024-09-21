"""Tests for the transforms module."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike

import interpies
from interpies import spatial
from interpies.grid import Grid
from interpies.transforms import find_trend, simple_resample

DATA_FOLDER = Path(__file__).parents[1] / 'data'


@pytest.fixture
def brtpgrd() -> Grid:
    return interpies.open(DATA_FOLDER / 'brtpgrd.gxf')


@pytest.fixture
def brtpgrd_res4(brtpgrd) -> ArrayLike:
    return simple_resample(brtpgrd.data, sampling=4)


def test_find_trend(brtpgrd_res4):
    nr, nc = brtpgrd_res4.shape
    xll = 50
    yll = 50
    cellsize = 100
    point_coords = spatial.grid_to_points(xll, yll, cellsize, nr, nc, flipy=True)
    trend = find_trend(point_coords, brtpgrd_res4, degree=1, returnModel=False)
    model = find_trend(point_coords, brtpgrd_res4, degree=1, returnModel=True)
    assert trend[0, 0] == pytest.approx(-522.0573457097089)  # type: ignore
    assert model.predict(np.arange(10).reshape(-1, 2)) == pytest.approx(
        [-438.60167674, -438.59546518, -438.58925362, -438.58304205, -438.57683049]
    )
