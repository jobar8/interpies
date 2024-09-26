"""Tests for the spatial module."""

from numpy.testing import assert_allclose

from interpies.spatial import project_points


def test_project_points():
    projected = project_points([[45.0, -6], [45.0, -9], [45.0, -12.0], [60, 0.0]], s_srs=4326, t_srs=23029)
    desired = [
        [736557.69131476, 4987548.4788836],
        [500108.90395927, 4983169.23651159],
        [263542.99025614, 4987422.43017289],
        [1001073.0017133, 6685825.55236939],
    ]
    assert_allclose(projected, desired)

    unprojected = project_points(
        desired,
        s_srs=23029,
        t_srs=4326,
    )
    assert_allclose(unprojected, [[45.0, -6], [45.0, -9], [45.0, -12.0], [60, 0.0]], atol=1e-6)
