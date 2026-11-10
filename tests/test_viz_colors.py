import numpy as np

from tractwise.viz.colors import categorical_palette, directional_rgb


def test_directional_rgb_axis_aligned():
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float),  # L-R -> red
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], float),  # S-I -> blue
    ]
    colors = directional_rgb(lines)
    assert colors.shape == (6, 3)
    np.testing.assert_allclose(colors[:3], [[1, 0, 0]] * 3)
    np.testing.assert_allclose(colors[3:], [[0, 0, 1]] * 3)


def test_directional_rgb_is_orientation_invariant():
    line = np.cumsum(np.random.default_rng(0).normal(size=(10, 3)), axis=0)
    np.testing.assert_allclose(
        directional_rgb([line]), directional_rgb([line[::-1]])[::-1], atol=1e-6
    )


def test_categorical_palette():
    pal = categorical_palette(12)
    assert pal.shape == (12, 3)
    assert pal.min() >= 0 and pal.max() <= 1
    assert len(np.unique(pal.round(3), axis=0)) == 12
