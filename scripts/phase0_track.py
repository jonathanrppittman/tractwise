"""Phase 0: whole-brain deterministic tractography on DIPY's Stanford HARDI.

Usage:
    uv run python scripts/phase0_track.py [--out outputs/phase0/stanford_det.trk]
"""

import argparse
from pathlib import Path

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines, length
from dipy.tracking.tracker import deterministic_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/phase0/stanford_det.trk")
    )
    parser.add_argument(
        "--density", type=int, default=2, help="seeds per voxel per axis"
    )
    parser.add_argument("--min-length", type=float, default=30.0, help="mm")
    parser.add_argument("--gfa-thr", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    hardi_fname, hardi_bval, hardi_bvec = get_fnames(name="stanford_hardi")
    label_fname = get_fnames(name="stanford_labels")
    data, affine, img = load_nifti(hardi_fname, return_img=True)
    labels = load_nifti(label_fname)[0]
    bvals, bvecs = read_bvals_bvecs(hardi_bval, hardi_bvec)
    gtab = gradient_table(bvals, bvecs=bvecs)

    # Stanford labels: 1 = white matter, 2 = corpus callosum seed ROI.
    white_matter = (labels == 1) | (labels == 2)
    csa_model = CsaOdfModel(gtab, sh_order_max=6)
    print("Fitting CSA model and extracting peaks...")
    peaks = peaks_from_model(
        csa_model,
        data,
        default_sphere,
        relative_peak_threshold=0.8,
        min_separation_angle=45,
        mask=white_matter,
    )
    save_nifti(args.out.parent / "gfa.nii.gz", peaks.gfa.astype(np.float32), affine)

    stopping = ThresholdStoppingCriterion(peaks.gfa, args.gfa_thr)
    seeds = utils.seeds_from_mask(white_matter, affine, density=args.density)
    print(f"Tracking from {len(seeds)} seeds...")
    streamlines = deterministic_tracking(
        seeds,
        stopping,
        affine,
        sh=peaks.shm_coeff,
        random_seed=args.seed,
        max_angle=30,
        step_size=0.5,
    )
    streamlines = Streamlines(streamlines)
    keep = length(streamlines) >= args.min_length
    streamlines = streamlines[keep]
    print(f"Kept {len(streamlines)} streamlines >= {args.min_length} mm")

    sft = StatefulTractogram(streamlines, img, Space.RASMM)
    save_tractogram(sft, str(args.out), bbox_valid_check=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
