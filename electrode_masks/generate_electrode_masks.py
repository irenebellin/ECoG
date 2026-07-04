#!/usr/bin/env python3
"""
generate_electrode_masks.py
----------------------------
Generate binary NIfTI sphere masks from MNI coordinates stored in
mni_electrode_coordinates.pkl.

Each recording site gets one binary NIfTI mask: a sphere centred on
the contact's MNI coordinate, with radius:
  - 3 mm for sEEG depth contacts (AdTech RD10R, default)
  - 5 mm for subdural strip contacts (AdTech IS04R)

Masks are named <SUBJECT>_e<NNN>_mask.nii.gz where e001, e002, ...
follow the same order as the keys in each subject's OrderedDict in
the pickle file.

Output layout:
    output_dir/
        SD010/
            SD010_e001_mask.nii.gz
            SD010_e002_mask.nii.gz
            ...

Dependencies:
    pip install nibabel numpy

Usage:
    python generate_electrode_masks.py \\
        --coords mni_electrode_coordinates.pkl \\
        --out_dir electrode_masks \\
        --template $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz

    # Process specific subjects only:
    python generate_electrode_masks.py \\
        --coords mni_electrode_coordinates.pkl \\
        --out_dir electrode_masks \\
        --template $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz \\
        --subjects SD010 SD011 SD013

    # Use 5mm radius (subdural strip contacts):
    python generate_electrode_masks.py \\
        --coords mni_electrode_coordinates.pkl \\
        --out_dir electrode_masks \\
        --template $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz \\
        --radius 5
"""

import argparse
import os
import pickle
import sys
import numpy as np
import nibabel as nib


# =============================================================================
# MASK GENERATION
# =============================================================================

def get_reference_image(template_path):
    if not os.path.exists(template_path):
        raise RuntimeError(
            f"MNI152 template not found: {template_path}\n"
            f"Check that FSLDIR is set and the template exists."
        )
    return nib.load(template_path)


def mni_to_voxel(coord_mm, affine):
    """Convert MNI mm coordinate to voxel index via inverse affine."""
    inv_affine = np.linalg.inv(affine)
    coord_h    = np.array([coord_mm[0], coord_mm[1], coord_mm[2], 1.0])
    vox        = inv_affine @ coord_h
    return np.round(vox[:3]).astype(int)


def make_sphere_mask(shape, center_vox, radius_mm, affine):
    """Create a binary sphere mask in voxel space, using mm distances."""
    i_ax = np.arange(shape[0])
    j_ax = np.arange(shape[1])
    k_ax = np.arange(shape[2])
    I, J, K = np.meshgrid(i_ax, j_ax, k_ax, indexing="ij")

    vox_coords   = np.column_stack([I.ravel(), J.ravel(), K.ravel()])
    vox_coords_h = np.column_stack([vox_coords, np.ones(len(vox_coords))])
    mm_coords    = (affine @ vox_coords_h.T).T[:, :3]
    center_mm    = (affine @ np.append(center_vox, 1))[:3]
    dist_mm      = np.sqrt(((mm_coords - center_mm) ** 2).sum(axis=1))

    return (dist_mm <= radius_mm).reshape(shape)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", required=True, type=str,
                     help="Path to mni_electrode_coordinates.pkl")
    ap.add_argument("--out_dir", required=True, type=str,
                     help="Output directory for electrode masks")
    ap.add_argument("--template", required=True, type=str,
                     help="Path to MNI152 1mm NIfTI template "
                          "(e.g. $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz)")
    ap.add_argument("--subjects", nargs="*", default=None,
                     help="Subject IDs to process (default: all in pickle)")
    ap.add_argument("--radius", type=float, default=3.0,
                     help="Sphere radius in mm (default: 3mm for sEEG depth contacts; "
                          "use 5mm for subdural strip contacts)")
    args = ap.parse_args()

    print(f"Loading coordinates from {args.coords}")
    with open(args.coords, "rb") as f:
        data = pickle.load(f)

    print(f"Loading MNI152 template from {args.template}")
    ref_img = get_reference_image(args.template)
    affine  = ref_img.affine
    shape   = ref_img.shape[:3]

    subjects = args.subjects if args.subjects else list(data.keys())

    for subj in subjects:
        if subj not in data:
            print(f"\n[WARN] {subj} not found in pickle, skipping.")
            continue

        chans = data[subj]  # OrderedDict: site_name -> [x, y, z]

        print(f"\n{'='*60}")
        print(f"Subject: {subj}  |  {len(chans)} contacts  |  radius={args.radius}mm")

        subj_out = os.path.join(args.out_dir, subj)
        os.makedirs(subj_out, exist_ok=True)

        all_masks = []

        for idx, (site_name, coord) in enumerate(chans.items(), start=1):
            elec_label = f"e{idx:03d}"
            coord      = np.array(coord)
            center_vox = mni_to_voxel(coord, affine)

            if any(center_vox < 0) or any(center_vox >= np.array(shape)):
                print(f"  [WARN] {elec_label} ({site_name}) at MNI {np.round(coord,2)} "
                      f"-> voxel {center_vox} is OUTSIDE volume, skipping.")
                continue

            mask    = make_sphere_mask(shape, center_vox, args.radius, affine)
            mask_u8 = mask.astype(np.uint8)
            all_masks.append(mask_u8)

            mask_img = nib.Nifti1Image(mask_u8, affine, ref_img.header)
            mask_img.header.set_data_dtype(np.uint8)
            out_path = os.path.join(subj_out, f"{subj}_{elec_label}_mask.nii.gz")
            nib.save(mask_img, out_path)
            print(f"  [{elec_label}] {site_name:>8}  MNI {np.round(coord,2)} "
                  f"-> voxel {center_vox}")

        # 4D combined mask (all contacts stacked)
        if all_masks:
            combined     = np.stack(all_masks, axis=-1)
            combined_img = nib.Nifti1Image(combined, affine, ref_img.header)
            combined_img.header.set_data_dtype(np.uint8)
            combined_path = os.path.join(subj_out,
                                          f"{subj}_all_electrodes_4D_mask.nii.gz")
            nib.save(combined_img, combined_path)
            print(f"\n  4D mask saved: {combined_path}")
            print(f"  Done: {len(all_masks)} individual masks + 1 combined 4D mask.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
