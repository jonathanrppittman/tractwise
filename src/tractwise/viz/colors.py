"""Streamline color schemes."""

import colorsys

import numpy as np


def directional_rgb(streamlines) -> np.ndarray:
    """Per-vertex directional colors (R = L-R, G = A-P, B = S-I).

    Each vertex is colored by the absolute value of its unit tangent, estimated
    with central differences. Returns a flat ``(total_vertices, 3)`` float32
    array in the same order as the concatenated streamline points.
    """
    colors = []
    for sl in streamlines:
        sl = np.asarray(sl, dtype=np.float32)
        if len(sl) < 2:
            colors.append(np.ones((len(sl), 3), dtype=np.float32))
            continue
        tangent = np.gradient(sl, axis=0)
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        colors.append(np.abs(tangent) / np.maximum(norm, 1e-8))
    return np.concatenate(colors).astype(np.float32)


def categorical_palette(n: int, *, seed: int = 0) -> np.ndarray:
    """``n`` well-separated, saturated RGB colors for labels or clusters."""
    offset = np.random.default_rng(seed).random()
    golden = 0.618033988749895  # golden-ratio hue spacing avoids neighbours clashing
    return np.array(
        [
            colorsys.hsv_to_rgb(
                (offset + golden * k) % 1.0,
                0.85 if k % 2 == 0 else 0.65,
                1.0 if k % 3 == 0 else 0.9,
            )
            for k in range(n)
        ],
        dtype=np.float32,
    )
