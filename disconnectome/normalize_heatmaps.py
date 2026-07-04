#!/usr/bin/env python3
"""
normalize_heatmaps.py

Linearly normalizes disconnectome heatmaps to [0, 1] range by rescaling
nonzero voxel values from their observed minimum to their observed maximum.
Zero voxels remain zero (transparent in visualization tools).

Optionally applies a mild Gaussian spatial smoothing kernel before
normalization to reduce isolated voxel scatter.

Run:
    pip install nibabel numpy scipy --break-system-packages
    python normalize_heatmaps.py \\
        --heatmap_root /path/to/heatmaps \\
        --out_root     /path/to/heatmaps_normalized \\
        --subjects SD010 SD011 SD013 SD015 SD017 SD018 SD019 SD021 SD022 \\
        --smooth_mm 0.8
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("pip install nibabel --break-system-packages")

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def normalize(nii_path, out_path, smooth_mm=0.0):
    img  = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)

    if smooth_mm > 0:
        if not HAS_SCIPY:
            print("  [WARN] scipy not available -- skipping smoothing")
        else:
            voxel_size = float(np.abs(img.affine[0, 0]))
            sigma = smooth_mm / voxel_size
            data = gaussian_filter(data, sigma=sigma)
            print(f"    smoothed: sigma={smooth_mm}mm ({sigma:.2f} vox)")

    nonzero = data[data > 0]
    if nonzero.size == 0:
        print(f"  [SKIP] {nii_path.name} -- no nonzero voxels")
        return False

    vmin = float(nonzero.min())
    vmax = float(nonzero.max())
    print(f"  {nii_path.name}: {vmin:.4g} -> {vmax:.4g} rescaled to 0..1")

    out = np.zeros_like(data)
    mask = data > 0
    out[mask] = (data[mask] - vmin) / max(vmax - vmin, 1e-9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, img.affine, img.header), str(out_path))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heatmap_root", required=True, type=Path,
                     help="Root folder containing grand_average/, per_subject/, per_electrode/ subfolders")
    ap.add_argument("--out_root", required=True, type=Path,
                     help="Output root folder (same subfolder structure)")
    ap.add_argument("--subjects", nargs="+", required=True,
                     help="Subject IDs to process")
    ap.add_argument("--smooth_mm", type=float, default=0.0,
                     help="Gaussian smoothing kernel sigma in mm (default: 0 = no smoothing)")
    args = ap.parse_args()

    print("--- Grand average ---")
    normalize(
        args.heatmap_root / "grand_average" / "grand_average_heatmap.nii.gz",
        args.out_root / "grand_average" / "grand_average_heatmap.nii.gz",
        args.smooth_mm
    )

    print("\n--- Per-subject ---")
    for subj in args.subjects:
        normalize(
            args.heatmap_root / "per_subject" / f"{subj}_heatmap.nii.gz",
            args.out_root / "per_subject" / f"{subj}_heatmap.nii.gz",
            args.smooth_mm
        )

    print("\n--- Per-electrode ---")
    for subj in args.subjects:
        subj_dir = args.heatmap_root / "per_electrode" / subj
        if not subj_dir.exists():
            continue
        for f in sorted(subj_dir.glob("*_heatmap.nii.gz")):
            normalize(f, args.out_root / "per_electrode" / subj / f.name,
                      args.smooth_mm)

    print(f"\nDone. Normalized heatmaps saved to {args.out_root}")


if __name__ == "__main__":
    main()
