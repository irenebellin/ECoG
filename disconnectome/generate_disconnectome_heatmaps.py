#!/usr/bin/env python3
"""
generate_disconnectome_heatmaps.py

Builds three levels of disconnectome heatmaps from per-recording-site
disconnectome masks (from BCBToolkit outputs, one .nii.gz per ECoG
contact):

    1. Per-electrode heatmaps  (average across all contacts of one shaft,
       e.g. all RA1..RA10 -> RA.nii.gz)
    2. Per-subject heatmaps    (average across all electrodes/contacts of
       one patient -> SD010.nii.gz)
    3. Grand-average heatmap   (average across all subjects -> grand_average.nii.gz)

It uses the recording-site -> electrode mapping derived from
mni_electrode_coordinates.pkl (channel name = electrode label + contact
number, e.g. "RA3" -> electrode "RA", contact 3), and assumes mask files
are named "<SUBJ>_e<NNN>_mask.nii.gz" where e001, e002, ... follow the
SAME ORDER as the keys in the subject's OrderedDict in the coordinates
pickle.

If your mask filenames already encode the site name directly (e.g.
SD010_RA3_mask.nii.gz) just adjust SITE_NAME_FROM_FILENAME below.

USAGE
-----
python generate_disconnectome_heatmaps.py \
    --coords mni_electrode_coordinates.pkl \
    --masks_dir /path/to/disconnectome \
    --out_dir   /path/to/disconnectome_heatmaps \
    --mode mean        # or "sum"

Expected input layout (one folder per subject, single contact sites masks directly inside):

    masks_dir/
        SD010/
            SD010_e001_mask.nii.gz
            SD010_e002_mask.nii.gz
            ...
        SD011/
            SD011_e001_mask.nii.gz
            ...

Output layout:

    out_dir/
        per_electrode/
            SD010/
                SD010_RA_heatmap.nii.gz
                SD010_RHH_heatmap.nii.gz
                ...
            SD011/
                ...
        per_subject/
            SD010_heatmap.nii.gz
            SD011_heatmap.nii.gz
            ...
        grand_average/
            grand_average_heatmap.nii.gz
        electrode_site_mapping.csv      <- audit trail of every file used

Requires: nibabel, numpy  (pip install nibabel numpy --break-system-packages)
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
    sys.exit("This script requires nibabel. Install with:\n"
              "  pip install nibabel --break-system-packages")


# --------------------------------------------------------------------------
# 1. Channel-name parsing (electrode label + contact number)
# --------------------------------------------------------------------------

def split_label(key):
    """'RA3' -> ('RA', 3). Falls back to (key, None) if no trailing digits."""
    m = re.match(r'^([A-Za-z]+)(\d+)$', key)
    return (m.group(1), int(m.group(2))) if m else (key, None)


def assign_electrode(subj, key):
    """
    Returns (electrode_label, contact_number) for a given site key.

    Special-cased for SD017's 'RD' family, which bundles 4 separate depth
    electrodes (RD1-RD4, 10 contacts each) under names like 'RD11'..'RD110',
    'RD21'..'RD210', etc. (electrode index digit immediately after 'RD',
    contact number follows). Add further special cases here if other
    subjects/datasets turn out to have similar ambiguous naming -- check
    by looking for any label group with an unexpectedly large contact
    count or non-monotonic/non-collinear coordinates (see the spatial
    sanity check this script also runs).
    """
    label, num = split_label(key)
    if subj == 'SD017' and label == 'RD':
        s = str(num)
        if len(s) == 3:  # e.g. '110' -> electrode 1, contact 10
            elec_idx, contact = s[0], int(s[1:])
        else:  # 2-digit, e.g. '11' -> electrode 1, contact 1
            elec_idx, contact = s[0], int(s[1])
        return f"RD{elec_idx}", contact
    return label, num


# --------------------------------------------------------------------------
# 2. Build site -> electrode mapping + site -> mask filename mapping
# --------------------------------------------------------------------------

def build_site_mapping(coords_pkl):
    """
    Returns: dict subj -> list of (site_name, electrode_label, contact_num, coord)
             IN THE SAME ORDER as the OrderedDict in the pickle (this order
             is what e001, e002, ... is assumed to follow).
    """
    with open(coords_pkl, 'rb') as f:
        data = pickle.load(f)

    mapping = {}
    for subj, chans in data.items():
        rows = []
        for site_name, coord in chans.items():
            elec, contact = assign_electrode(subj, site_name)
            rows.append((site_name, elec, contact, np.asarray(coord)))
        mapping[subj] = rows
    return mapping


def mask_filename_for_index(subj, idx):
    """e.g. subj='SD010', idx=1 -> 'SD010_e001_mask.nii.gz'"""
    return f"{subj}_e{idx:03d}_mask.nii.gz"


# If your masks are instead already named by site (e.g. SD010_RA3_mask.nii.gz),
# switch to something like this and use it instead of mask_filename_for_index:
#
# def mask_filename_for_site(subj, site_name):
#     return f"{subj}_{site_name}_mask.nii.gz"


# --------------------------------------------------------------------------
# 3. Spatial sanity check -- flags electrode
#    groups whose contacts aren't roughly collinear / evenly spaced, which
#    would suggest the e00N <-> site ordering assumption is wrong for that
#    subject and should be checked manually before trusting the heatmaps.
# --------------------------------------------------------------------------

def sanity_check_electrode_geometry(mapping):
    print("\n--- Spatial sanity check (electrode label parsing) ---")
    any_flag = False
    for subj, rows in mapping.items():
        by_elec = {}
        for site_name, elec, contact, coord in rows:
            by_elec.setdefault(elec, []).append((contact, coord))
        for elec, items in by_elec.items():
            items.sort(key=lambda x: x[0])
            coords = np.array([c for _, c in items])
            if len(coords) < 2:
                continue
            diffs = np.diff(coords, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            v = coords[-1] - coords[0]
            norm = np.linalg.norm(v)
            if norm == 0:
                continue
            v = v / norm
            maxdev = max(
                np.linalg.norm((c - coords[0]) - np.dot(c - coords[0], v) * v)
                for c in coords
            )
            if dists.std() > 2.5 or maxdev > 5:
                print(f"  CHECK  {subj} {elec}: n={len(items)} "
                      f"spacing={dists.mean():.2f}+/-{dists.std():.2f}mm "
                      f"max_deviation_from_line={maxdev:.2f}mm")
                any_flag = True
    if not any_flag:
        print("  All electrode groups look spatially consistent.")
    print("--------------------------------------------------------\n")


# --------------------------------------------------------------------------
# 4. Core averaging / summing logic
# --------------------------------------------------------------------------

def load_mask(path):
    img = nib.load(str(path))
    return img, img.get_fdata(dtype=np.float32)


def combine_volumes(paths, mode):
    """Load a list of NIfTI paths and sum or average them. Returns
    (combined_array, reference_nifti_image, n_loaded, missing_paths)."""
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
        acc += arr
        n_loaded += 1
    if n_loaded == 0:
        return None, None, 0, missing
    if mode == "mean":
        acc = acc / n_loaded
    return acc, ref_img, n_loaded, missing


def save_volume(arr, ref_img, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(out_img, str(out_path))


# --------------------------------------------------------------------------
# 5. Main pipeline
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", required=True, type=Path,
                     help="Path to mni_electrode_coordinates.pkl")
    ap.add_argument("--masks_dir", required=True, type=Path,
                     help="Folder containing one subfolder per subject with mask .nii.gz files")
    ap.add_argument("--out_dir", required=True, type=Path,
                     help="Output folder for the organized heatmaps")
    ap.add_argument("--mode", choices=["mean", "sum"], default="mean",
                     help="Combine masks by averaging (default) or summing")
    ap.add_argument("--subjects", nargs="*", default=None,
                     help="Optional: restrict to specific subject IDs (default: all in pickle)")
    ap.add_argument("--skip_sanity_check", action="store_true",
                     help="Skip the electrode-geometry spatial sanity check")
    args = ap.parse_args()

    mapping = build_site_mapping(args.coords)

    if args.subjects:
        mapping = {s: mapping[s] for s in args.subjects if s in mapping}

    if not args.skip_sanity_check:
        sanity_check_electrode_geometry(mapping)

    per_electrode_dir = args.out_dir / "per_electrode"
    per_subject_dir = args.out_dir / "per_subject"
    grand_average_dir = args.out_dir / "grand_average"
    for d in (per_electrode_dir, per_subject_dir, grand_average_dir):
        d.mkdir(parents=True, exist_ok=True)

    audit_rows = []  # for electrode_site_mapping.csv

    subject_level_volumes = []  # (subj_array, ref_img) for grand average
    subject_level_weights = []  # n masks per subject, for weighted grand average

    for subj, rows in mapping.items():
        subj_dir_in = args.masks_dir / subj
        if not subj_dir_in.exists():
            print(f"[WARN] No mask folder found for {subj} at {subj_dir_in}, skipping subject.")
            continue

        # site_name -> mask path, using e00N order = OrderedDict order
        site_to_path = {}
        for idx, (site_name, elec, contact, coord) in enumerate(rows, start=1):
            site_to_path[site_name] = subj_dir_in / mask_filename_for_index(subj, idx)

        # ---- per-electrode heatmaps ----
        by_elec = {}
        for site_name, elec, contact, coord in rows:
            by_elec.setdefault(elec, []).append(site_name)

        all_subject_mask_paths = []
        for elec, site_names in by_elec.items():
            paths = [site_to_path[s] for s in site_names]
            all_subject_mask_paths.extend(paths)
            combined, ref_img, n_loaded, missing = combine_volumes(paths, args.mode)

            for s, p in zip(site_names, paths):
                audit_rows.append([subj, elec, s, str(p), p.exists()])

            if combined is None:
                print(f"[WARN] {subj} electrode {elec}: no mask files found, skipping.")
                continue
            if missing:
                print(f"[WARN] {subj} electrode {elec}: missing {len(missing)}/{len(paths)} "
                      f"mask files (used {n_loaded}).")

            out_path = per_electrode_dir / subj / f"{subj}_{elec}_heatmap.nii.gz"
            save_volume(combined, ref_img, out_path)

        # ---- per-subject heatmap (across ALL electrodes/contacts of this subject) ----
        combined, ref_img, n_loaded, missing = combine_volumes(all_subject_mask_paths, args.mode)
        if combined is None:
            print(f"[WARN] {subj}: no mask files found at all, skipping subject-level heatmap.")
            continue
        if missing:
            print(f"[WARN] {subj}: missing {len(missing)}/{len(all_subject_mask_paths)} "
                  f"mask files overall (used {n_loaded}).")

        out_path = per_subject_dir / f"{subj}_heatmap.nii.gz"
        save_volume(combined, ref_img, out_path)
        print(f"[OK] {subj}: per-electrode ({len(by_elec)} electrodes) "
              f"+ per-subject heatmaps written ({n_loaded} masks used).")

        subject_level_volumes.append((combined, ref_img))
        subject_level_weights.append(n_loaded)

    # ---- grand average across subjects ----
    if subject_level_volumes:
        # Equal-weight average across subjects (each subject contributes
        # one already-averaged volume). Switch to a weighted average by
        # n masks if you'd rather weight by electrode coverage instead of
        # by subject.
        ref_img = subject_level_volumes[0][1]
        stack = np.stack([v for v, _ in subject_level_volumes], axis=0)
        grand = stack.mean(axis=0) if args.mode == "mean" else stack.sum(axis=0)
        out_path = grand_average_dir / "grand_average_heatmap.nii.gz"
        save_volume(grand, ref_img, out_path)
        print(f"\n[OK] Grand average heatmap written across {len(subject_level_volumes)} subjects.")
    else:
        print("\n[WARN] No subjects processed -- grand average not created.")

    # ---- audit trail ----
    audit_path = args.out_dir / "electrode_site_mapping.csv"
    with open(audit_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "electrode", "site_name", "mask_path", "mask_found"])
        w.writerows(audit_rows)
    print(f"[OK] Audit trail written to {audit_path}")

    print(f"\nAll done. Outputs organized under: {args.out_dir}")
    print(f"  {per_electrode_dir}/<SUBJ>/<SUBJ>_<ELEC>_heatmap.nii.gz")
    print(f"  {per_subject_dir}/<SUBJ>_heatmap.nii.gz")
    print(f"  {grand_average_dir}/grand_average_heatmap.nii.gz")


if __name__ == "__main__":
    main()
