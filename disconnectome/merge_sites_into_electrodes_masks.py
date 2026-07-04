#!/usr/bin/env python3
"""
merge_sites_into_electrode_masks.py

Merges the individual recording-site disconnectome masks belonging to the
same physical electrode (e.g. RA1..RA10) into a SINGLE NIfTI volume per
electrode (e.g. RA_merged.nii.gz).

Uses the same site -> electrode parsing logic established earlier (channel
name = electrode label + contact number, e.g. "RA3" -> electrode "RA",
contact 3; with the SD017 'RD1'-'RD4' special case handled).

Merge modes (--mode):
    union  (default) - binary union: voxel = 1 if ANY site's mask is
                        nonzero there, else 0. Best if your masks are
                        already binary disconnectome masks and you just
                        want "the combined territory disconnected by this
                        electrode."
    sum               - voxelwise sum across all sites of the electrode.
    mean              - voxelwise average across all sites of the electrode.

USAGE
-----
python merge_sites_into_electrode_masks.py \
    --coords mni_electrode_coordinates.pkl \
    --masks_dir /path/to/disconnectome \
    --out_dir   /path/to/electrode_merged_masks \
    --mode union

Expected input layout (one folder per subject, masks directly inside,
named "<SUBJ>_e<NNN>_mask.nii.gz" where e001, e002, ... follow the SAME
ORDER as the keys in that subject's OrderedDict in the coordinates pickle):

    masks_dir/
        SD010/
            SD010_e001_mask.nii.gz
            SD010_e002_mask.nii.gz
            ...

Output:
    out_dir/
        SD010/
            SD010_RA_merged.nii.gz
            SD010_RHH_merged.nii.gz
            ...
        SD011/
            ...
        merge_audit.csv   <- which mask files went into which merged electrode

Requires: nibabel, numpy (pip install nibabel numpy --break-system-packages)
"""

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("Requires nibabel. Install with: pip install nibabel --break-system-packages")


# --------------------------------------------------------------------------
# Channel-name parsing (same logic as generate_disconnectome_heatmaps.py)
# --------------------------------------------------------------------------

def split_label(key):
    """'RA3' -> ('RA', 3). Falls back to (key, None) if no trailing digits."""
    m = re.match(r'^([A-Za-z]+)(\d+)$', key)
    return (m.group(1), int(m.group(2))) if m else (key, None)


def assign_electrode(subj, key):
    """
    Returns (electrode_label, contact_number) for a given site key.

    Special-cased for SD017's 'RD' family (RD1-RD4, 10 contacts each,
    named 'RD11'..'RD110', 'RD21'..'RD210', etc.). Add further special
    cases here if other subjects turn out to have similarly ambiguous
    naming -- check by looking for an unexpectedly large contact count or
    non-monotonic/non-collinear coordinates for a label group.
    """
    label, num = split_label(key)
    if subj == 'SD017' and label == 'RD':
        s = str(num)
        if len(s) == 3:
            elec_idx, contact = s[0], int(s[1:])
        else:
            elec_idx, contact = s[0], int(s[1])
        return f"RD{elec_idx}", contact
    return label, num


def build_site_mapping(coords_pkl):
    """Returns subj -> list of (site_name, electrode_label, contact_num),
    in the same order as the OrderedDict in the pickle."""
    with open(coords_pkl, 'rb') as f:
        data = pickle.load(f)

    mapping = {}
    for subj, chans in data.items():
        rows = []
        for site_name in chans.keys():
            elec, contact = assign_electrode(subj, site_name)
            rows.append((site_name, elec, contact))
        mapping[subj] = rows
    return mapping


def mask_filename_for_index(subj, idx):
    return f"{subj}_e{idx:03d}_mask.nii.gz"


# --------------------------------------------------------------------------
# Merge logic
# --------------------------------------------------------------------------

def load_mask(path):
    img = nib.load(str(path))
    return img, img.get_fdata(dtype=np.float32)


def merge_paths(paths, mode):
    """Load and merge a list of NIfTI paths. Returns (merged_array,
    reference_image, n_loaded, missing_paths)."""
    ref_img = None
    acc = None
    n_loaded = 0
    missing = []
    for p in paths:
        if not p.exists():
            missing.append(p)
            continue
        img, arr = load_mask(p)
        if ref_img is None:
            ref_img = img
            acc = np.zeros_like(arr, dtype=np.float64)
        if mode == "union":
            acc = np.maximum(acc, (arr != 0).astype(np.float64))
        else:
            acc += arr
        n_loaded += 1
    if n_loaded == 0:
        return None, None, 0, missing
    if mode == "mean":
        acc = acc / n_loaded
    return acc, ref_img, n_loaded, missing


def save_volume(arr, ref_img, out_path, mode):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if mode == "union" else np.float32
    out_img = nib.Nifti1Image(arr.astype(dtype), ref_img.affine, ref_img.header)
    nib.save(out_img, str(out_path))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", required=True, type=Path)
    ap.add_argument("--masks_dir", required=True, type=Path,
                     help="Parent folder containing one subfolder per subject")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--mode", choices=["union", "sum", "mean"], default="union",
                     help="How to combine the per-site masks into one electrode "
                          "volume (default: union -- binary OR across sites)")
    ap.add_argument("--subjects", nargs="*", default=None,
                     help="Optional: restrict to specific subject IDs (default: all)")
    args = ap.parse_args()

    mapping = build_site_mapping(args.coords)
    if args.subjects:
        mapping = {s: mapping[s] for s in args.subjects if s in mapping}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []

    for subj, rows in mapping.items():
        subj_dir_in = args.masks_dir / subj
        if not subj_dir_in.exists():
            print(f"[WARN] No mask folder for {subj} at {subj_dir_in}, skipping.")
            continue

        site_to_path = {
            site_name: subj_dir_in / mask_filename_for_index(subj, idx)
            for idx, (site_name, elec, contact) in enumerate(rows, start=1)
        }

        by_elec = {}
        for site_name, elec, contact in rows:
            by_elec.setdefault(elec, []).append(site_name)

        for elec, site_names in by_elec.items():
            paths = [site_to_path[s] for s in site_names]
            merged, ref_img, n_loaded, missing = merge_paths(paths, args.mode)

            for s, p in zip(site_names, paths):
                audit_rows.append([subj, elec, s, str(p), p.exists()])

            if merged is None:
                print(f"[WARN] {subj} electrode {elec}: no mask files found, skipping.")
                continue
            if missing:
                print(f"[WARN] {subj} electrode {elec}: missing {len(missing)}/{len(paths)} "
                      f"mask files (used {n_loaded}).")

            out_path = args.out_dir / subj / f"{subj}_{elec}_merged.nii.gz"
            save_volume(merged, ref_img, out_path, args.mode)
            print(f"[OK] {subj} {elec}: merged {n_loaded} sites -> {out_path}")

    audit_path = args.out_dir / "merge_audit.csv"
    with open(audit_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "electrode", "site_name", "mask_path", "mask_found"])
        w.writerows(audit_rows)
    print(f"\n[OK] Audit trail written to {audit_path}")
    print(f"All done. Merged electrode masks under: {args.out_dir}")


if __name__ == "__main__":
    main()
