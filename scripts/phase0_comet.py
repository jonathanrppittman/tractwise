"""Phase 0 prototype: "comet" animation over a streamline subset.

Bright comet heads with fading tails travel along each streamline with random
phase offsets and directions. Colors come from a QuickBundles clustering as a
stand-in for predicted bundle labels. Note that diffusion MRI gives fiber
orientation, not direction: the motion is a visualization device only.

Usage:
    uv run python scripts/phase0_comet.py [--trk outputs/phase0/stanford_det.trk]
"""

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.tracking.streamline import length, set_number_of_points
from fury import actor

from tractwise.viz.colors import categorical_palette
from tractwise.viz.render import offscreen_manager, render_frame, set_view


def comet_intensity(arclen, lengths, t, phase, direction, *, speed, tail, gap):
    """Per-vertex comet brightness in [0, 1] at time ``t`` (seconds).

    ``arclen`` is (S, N) distance in mm of each vertex from the streamline start.
    Each comet's head travels at ``speed`` mm/s and loops with a ``gap`` mm pause
    so short and long streamlines carry comets at the same physical speed.
    """
    s = np.where(direction[:, None] > 0, arclen, lengths[:, None] - arclen)
    period = lengths + gap
    head = (phase * period + speed * t) % period
    behind = head[:, None] - s  # distance behind the head (>0 = in the tail)
    return np.where(
        behind >= 0,
        np.exp(-behind / tail),
        np.exp(-((behind / (0.25 * tail)) ** 2)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trk", type=Path, default=Path("outputs/phase0/stanford_det.trk")
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/phase0/comet.mp4"))
    parser.add_argument("--n-streamlines", type=int, default=3000)
    parser.add_argument("--min-length", type=float, default=40.0, help="mm")
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--qb-threshold", type=float, default=20.0, help="mm")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--speed", type=float, default=40.0, help="comet speed, mm/s")
    parser.add_argument("--tail", type=float, default=20.0, help="tail length, mm")
    parser.add_argument(
        "--gap", type=float, default=60.0, help="pause between laps, mm"
    )
    parser.add_argument(
        "--turns", type=float, default=0.5, help="turntable revolutions"
    )
    parser.add_argument("--zoom", type=float, default=1.3)
    parser.add_argument("--gif", action="store_true", help="also write a small GIF")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    sft = load_tractogram(str(args.trk), "same", bbox_valid_check=False)
    streamlines = sft.streamlines[length(sft.streamlines) >= args.min_length]
    idx = rng.choice(
        len(streamlines), min(args.n_streamlines, len(streamlines)), replace=False
    )
    lines = np.asarray(
        set_number_of_points(streamlines[np.sort(idx)], nb_points=args.n_points)
    )
    n_lines, n_pts = lines.shape[:2]

    clusters = QuickBundles(threshold=args.qb_threshold).cluster(list(lines))
    labels = np.empty(n_lines, dtype=int)
    for k, cluster in enumerate(clusters):
        labels[cluster.indices] = k
    base_rgb = categorical_palette(len(clusters), seed=args.seed)[labels]
    print(f"{n_lines} streamlines, {len(clusters)} QuickBundles clusters")

    seg = np.linalg.norm(np.diff(lines, axis=1), axis=2)
    arclen = np.concatenate([np.zeros((n_lines, 1)), np.cumsum(seg, axis=1)], axis=1)
    lengths = arclen[:, -1]
    phase = rng.random(n_lines)
    direction = rng.choice([-1, 1], n_lines)

    per_vertex_rgb = np.repeat(base_rgb[:, None, :], n_pts, axis=1).reshape(-1, 3)
    glow = actor.streamlines(
        list(lines),
        colors=per_vertex_rgb * 0.14,
        thickness=1.0,
        opacity=1.0,
        outline_thickness=0.0,
        enable_picking=False,
    )
    comets = actor.streamlines(
        list(lines),
        colors=np.concatenate(
            [per_vertex_rgb, np.ones((len(per_vertex_rgb), 1))], axis=1
        ),
        thickness=3.5,
        opacity=1.0,
        outline_thickness=0.0,
        enable_picking=False,
    )
    comets.material.alpha_mode = "add"  # additive blending gives the glow
    comets.material.depth_write = False

    # The actor's buffer stores lines back to back with one NaN separator each.
    color_buf = comets.geometry.colors
    rgba = np.full((n_lines, n_pts + 1, 4), np.nan, dtype=np.float32)

    show_m = offscreen_manager([glow, comets], size=tuple(args.size))
    n_frames = int(round(args.seconds * args.fps))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    gif_frames = []
    t0 = time.perf_counter()
    for f in range(n_frames):
        t = f / args.fps
        inten = comet_intensity(
            arclen,
            lengths,
            t,
            phase,
            direction,
            speed=args.speed,
            tail=args.tail,
            gap=args.gap,
        )
        # Additive blending ignores alpha, so premultiply; the head runs white-hot.
        hot = inten[..., None] ** 4
        rgba[:, :n_pts, :3] = inten[..., None] * (
            (1 - hot) * base_rgb[:, None, :] + hot
        )
        rgba[:, :n_pts, 3] = inten
        color_buf.data[:] = rgba.reshape(-1, 4)[: len(color_buf.data)]
        color_buf.update_full()

        theta = 2 * np.pi * args.turns * f / n_frames + np.pi
        set_view(
            show_m, ((np.cos(theta), np.sin(theta), 0.35), (0, 0, 1)), zoom=args.zoom
        )
        frame = render_frame(show_m)[..., :3]
        writer.append_data(frame)
        if args.gif and f % 2 == 0:
            gif_frames.append(frame[::2, ::2])
        if f % 30 == 0:
            print(f"frame {f}/{n_frames} ({time.perf_counter() - t0:.0f} s)")
    writer.close()
    print(f"Saved {args.out}")
    if args.gif:
        gif_path = args.out.with_suffix(".gif")
        imageio.mimsave(gif_path, gif_frames, duration=2000 / args.fps, loop=0)
        print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
