"""Phase 0: render the whole-brain tractogram with directional RGB colors.

Usage:
    uv run python scripts/phase0_render.py [--trk outputs/phase0/stanford_det.trk]
"""

import argparse
import time
from pathlib import Path

import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamlinespeed import compress_streamlines
from fury import actor

from tractwise.viz.colors import directional_rgb
from tractwise.viz.render import offscreen_manager, render_frame, set_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trk", type=Path, default=Path("outputs/phase0/stanford_det.trk")
    )
    parser.add_argument("--outdir", type=Path, default=Path("outputs/phase0"))
    parser.add_argument("--max-streamlines", type=int, default=150_000)
    parser.add_argument("--size", type=int, nargs=2, default=(2400, 2400))
    parser.add_argument("--thickness", type=float, default=1.0)
    parser.add_argument("--opacity", type=float, default=0.6)
    parser.add_argument("--views", nargs="+", default=["axial", "sagittal", "oblique"])
    parser.add_argument("--zoom", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sft = load_tractogram(str(args.trk), "same", bbox_valid_check=False)
    streamlines = sft.streamlines
    if len(streamlines) > args.max_streamlines:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(streamlines), args.max_streamlines, replace=False))
        streamlines = streamlines[idx]
    print(f"Rendering {len(streamlines)} streamlines")

    # Compression keeps the vertex buffer under wgpu's 128 MB binding limit.
    lines = compress_streamlines(list(streamlines), tol_error=0.1)
    print(f"{sum(len(sl) for sl in lines)} vertices after compression")
    tracts = actor.streamlines(
        lines,
        colors=directional_rgb(lines),
        thickness=args.thickness,
        opacity=args.opacity,
        outline_thickness=0.0,
        enable_picking=False,
    )
    show_m = offscreen_manager([tracts], size=tuple(args.size))
    for view in args.views:
        set_view(show_m, view, zoom=args.zoom)
        fname = args.outdir / f"tractogram_{view}.png"
        t0 = time.perf_counter()
        render_frame(show_m, fname)
        print(f"Saved {fname} ({time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    main()
