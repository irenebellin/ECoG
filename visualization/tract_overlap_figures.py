#!/usr/bin/env python3
"""
tract_overlap_figures.py

For each disconnectome map (grand average, per-subject, per-electrode) and
each individual electrode site mask, computes the percentage overlap with
each tract in the Rojkova white matter atlas and generates a high-resolution
figure showing a horizontal bar chart of the results.

Overlap metric:
    For each atlas tract T (thresholded at >0.5) and binary map M (>0):
    overlap_pct = (voxels in both T and M) / (voxels in T) * 100

Only tracts with overlap > MIN_PCT are shown. Bars sorted by % descending.
Underscores in names are replaced with spaces.

Run:
    pip install nibabel numpy matplotlib --break-system-packages
    python tract_overlap_figures.py \\
        --atlas_dir   /path/to/Atlas_Rojkova \\
        --heatmap_root /path/to/heatmaps \\
        --masks_dir   /path/to/electrode_masks \\
        --out_root    /path/to/tract_overlap \\
        --subjects SD010 SD011 SD013 SD015 SD017 SD018 SD019 SD021 SD022 \\
        --min_pct 5.0
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import nibabel as nib
except ImportError:
    sys.exit("pip install nibabel --break-system-packages")

DPI       = 300
BG_COLOR  = "#FFFFFF"
TEXT_COLOR = "#1A1A2E"


def load_atlas_tracts(atlas_dir):
    tracts = []
    for f in sorted(atlas_dir.glob("*.nii.gz")):
        name   = f.stem.replace(".nii", "").replace("_", " ")
        img    = nib.load(str(f))
        data   = img.get_fdata(dtype=np.float32)
        binary = (data > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            continue
        tracts.append((name, binary))
    print(f"Loaded {len(tracts)} atlas tracts (thresholded at >0.5)")
    return tracts


def binarize(nii_path):
    img  = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    return (data > 0).astype(np.uint8)


def compute_overlap(binary_map, tracts, min_pct):
    results = []
    for tract_name, tract_binary in tracts:
        tract_voxels   = int(tract_binary.sum())
        overlap_voxels = int((binary_map & tract_binary).sum())
        if tract_voxels == 0:
            continue
        pct = overlap_voxels / tract_voxels * 100
        if pct > min_pct:
            results.append((tract_name, pct))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def make_figure(title, results, out_path):
    if not results:
        print(f"  [SKIP] No tracts above threshold for: {title}")
        return

    n     = len(results)
    names = [r[0] for r in results]
    pcts  = [r[1] for r in results]

    fig_h = max(3.5, 0.38 * n + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, fig_h), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y_pos      = np.arange(n)
    norm_pcts  = np.array(pcts) / max(pcts)
    colors     = [matplotlib.colormaps["Blues"](0.35 + 0.55 * p) for p in norm_pcts]

    bars = ax.barh(y_pos, pcts, color=colors, height=0.65,
                   edgecolor="white", linewidth=0.5)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="left",
                fontsize=8, color=TEXT_COLOR)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.5, color=TEXT_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("% of tract overlapping with map", fontsize=9,
                  color=TEXT_COLOR, labelpad=6)
    ax.set_xlim(0, min(100, max(pcts) * 1.18))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CCCCCC")
        ax.spines[spine].set_linewidth(0.7)

    ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=8)

    fig.suptitle(title.replace("_", " "), fontsize=12, fontweight="bold",
                 color=TEXT_COLOR, y=0.98, x=0.5, ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  saved {out_path.name}  ({n} tracts)")


def process_map(nii_path, title, out_path, tracts, min_pct):
    if not nii_path.exists():
        print(f"  [WARN] Not found: {nii_path}")
        return
    binary  = binarize(nii_path)
    results = compute_overlap(binary, tracts, min_pct)
    make_figure(title, results, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas_dir",    required=True, type=Path)
    ap.add_argument("--heatmap_root", required=True, type=Path)
    ap.add_argument("--masks_dir",    required=True, type=Path)
    ap.add_argument("--out_root",     required=True, type=Path)
    ap.add_argument("--subjects",     nargs="+", required=True)
    ap.add_argument("--min_pct",      type=float, default=5.0,
                     help="Minimum overlap %% to show in figures (default: 5)")
    args = ap.parse_args()

    tracts = load_atlas_tracts(args.atlas_dir)
    if not tracts:
        sys.exit(f"No .nii.gz files found in {args.atlas_dir}")

    print("\n--- Grand average ---")
    process_map(
        args.heatmap_root / "grand_average" / "grand_average_heatmap.nii.gz",
        "Grand average heatmap",
        args.out_root / "grand_average" / "grand_average_heatmap.png",
        tracts, args.min_pct
    )

    print("\n--- Per-subject ---")
    for subj in args.subjects:
        process_map(
            args.heatmap_root / "per_subject" / f"{subj}_heatmap.nii.gz",
            f"{subj} heatmap",
            args.out_root / "per_subject" / f"{subj}_heatmap.png",
            tracts, args.min_pct
        )

    print("\n--- Per-electrode ---")
    for subj in args.subjects:
        subj_dir = args.heatmap_root / "per_electrode" / subj
        if not subj_dir.exists():
            continue
        for f in sorted(subj_dir.glob("*_heatmap.nii.gz")):
            name = f.stem.replace(".nii", "")
            process_map(f, name.replace("_", " "),
                        args.out_root / "per_electrode" / subj / f"{name}.png",
                        tracts, args.min_pct)

    print("\n--- Individual electrode site masks ---")
    for subj in args.subjects:
        subj_mask_dir = args.masks_dir / subj
        if not subj_mask_dir.exists():
            continue
        for f in sorted(subj_mask_dir.glob(f"{subj}_e*_mask.nii.gz")):
            name = f.stem.replace(".nii", "")
            process_map(f, name.replace("_", " "),
                        args.out_root / "electrode_sites" / subj / f"{name}.png",
                        tracts, args.min_pct)

    print(f"\nAll done. Figures saved under {args.out_root}")


if __name__ == "__main__":
    main()
