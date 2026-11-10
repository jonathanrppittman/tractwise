"""Offscreen rendering helpers built on FURY 2.x (pygfx/wgpu).

FURY 2.x renders through wgpu, so no X server is needed: on a machine without a
GPU driver it falls back to Mesa's llvmpipe software renderer.
"""

from pathlib import Path

import numpy as np
from fury import window
from fury.window import render_screens

# Camera view directions in RAS+ world space (direction from target to camera).
VIEWS = {
    "axial": ((0, 0, 1), (0, 1, 0)),  # from above, anterior up
    "sagittal": ((-1, 0, 0), (0, 0, 1)),  # from the left, superior up
    "coronal": ((0, 1, 0), (0, 0, 1)),  # from the front, superior up
    "oblique": ((-1, 0.8, 0.6), (0, 0, 1)),  # 3/4 view from front-left-above
}


def offscreen_manager(actors, *, size=(1920, 1080), background=(0, 0, 0, 1)):
    """Create an offscreen ShowManager whose scene contains ``actors``."""
    scene = window.Scene(background=background)
    scene.add(*actors)
    return window.ShowManager(
        scene=scene, size=size, window_type="offscreen", pixel_ratio=1.0
    )


def set_view(show_m, view, *, zoom: float = 1.0):
    """Point the camera at the scene from a named view or a (dir, up) pair."""
    view_dir, up = VIEWS[view] if isinstance(view, str) else view
    screen = show_m.screens[0]
    camera = screen.camera
    # pygfx's view_dir points from the camera towards the target.
    camera.show_object(
        screen.scene.main_scene,
        view_dir=tuple(-np.asarray(view_dir, dtype=float)),
        up=up,
        scale=1.0 / zoom,
    )


def render_frame(show_m, fname: str | Path | None = None) -> np.ndarray:
    """Draw the scene once and return the RGBA frame (optionally saving it)."""
    render_screens(show_m.renderer, show_m.screens)
    return show_m.snapshot(fname=None if fname is None else str(fname))
