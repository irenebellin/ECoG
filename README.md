# ECoG Disconnectome Analysis

This repository contains the analysis pipeline used to generate and visualize disconnectome maps from intracranial ECoG electrode coordinates, as described in:

XXXX paper link XXXX 
---

## Overview

For each ECoG recording site, we generated a volumetric electrode mask in MNI152 space and computed a disconnectome map using the BCBToolkit — identifying which white matter tracts are in proximity to, and potentially modulated by, each electrode contact. Maps were aggregated at three levels (per recording site, per electrode shaft, per subject, and grand average across subjects) and visualized as 3D glass brain renders using MRIcroGL.

---

## Repository Structure

```
ECoG/
├── electrode_masks/
│   └── generate_electrode_masks.py     # generates binary NIfTI sphere masks from MNI coordinates
├── disconnectome/
│   ├── generate_disconnectome_heatmaps.py   # averages disconnectome maps per electrode/subject/group
│   ├── merge_sites_into_electrode_masks.py  # merges per-site masks into per-electrode NIfTIs
│   └── normalize_heatmaps.py               # linearly normalizes heatmaps to [0,1] range
├── visualization/
│   ├── mricroGL_glass_brain.py         # MRIcroGL scripting for 3D glass brain renders
│   ├── surfice_disconnectome.py        # Surfice scripting for surface-based renders
│   └── tract_overlap_figures.py        # computes overlap with Rojkova WM atlas, generates bar chart figures
└── methods/
    └── methods_disconnectome.md        # full methods section describing the pipeline
```

---

## Dependencies

### Python packages
```
pip install nibabel numpy matplotlib scipy
```

### External tools
- [BCBToolkit](https://www.bcblab.com) — disconnectome map computation
- [MRIcroGL](https://www.nitrc.org/projects/mricrogl) — 3D glass brain visualization
- [FSL](https://fsl.fmrib.ox.ac.uk) — MNI152 standard space template
- [Rojkova White Matter Atlas](https://doi.org/10.1093/brain/aww009) — tract overlap analysis

---

## Usage

### 1. Generate electrode masks
```bash
python electrode_masks/generate_electrode_masks.py
```
Generates one binary NIfTI sphere mask per recording site (radius: 3mm for sEEG depth contacts, 5mm for subdural strip contacts), centred on each contact's MNI coordinate.

### 2. Generate disconnectome heatmaps
After running BCBToolkit on the individual masks:
```bash
python disconnectome/generate_disconnectome_heatmaps.py \
    --coords mni_electrode_coordinates.pkl \
    --masks_dir /path/to/disconnectome \
    --out_dir /path/to/heatmaps \
    --mode mean
```
Outputs per-electrode, per-subject, and grand average heatmaps.

### 3. Normalize heatmaps
```bash
python disconnectome/normalize_heatmaps.py
```
Linearly rescales each heatmap to [0,1] for visualization.

### 4. Visualize in MRIcroGL
Open `visualization/mricroGL_glass_brain.py` in MRIcroGL via Scripting → Open → Ctrl+R.

### 5. Compute tract overlap
```bash
python visualization/tract_overlap_figures.py
```
Computes overlap between each heatmap and the Rojkova WM atlas tracts, generating publication-ready bar chart figures.

---

## Citation

If you use this pipeline, please cite:
- **BCBToolkit**: Foulon, C., Cerliani, L., Kinkingnéhun, S., Levy, R., Rosso, C., Urbanski, M., Volle, E., & Thiebaut de Schotten, M. (2018). Advanced lesion symptom mapping analyses and implementation as BCBtoolkit. *GigaScience*, 7(3), giy004. https://doi.org/10.1093/gigascience/giy004
- **Disconnectome approach**: Thiebaut de Schotten, M., Foulon, C., & Nachev, P. (2020). Brain disconnections link structural connectivity with function and behaviour. *Nature Communications*, 11, 5094. https://doi.org/10.1038/s41467-020-18920-9
- **Rojkova atlas**: Rojkova, K., Volle, E., Urbanski, M., Humbert, F., Dell'Acqua, F., & Thiebaut de Schotten, M. (2016). Atlasing the frontal lobe connections and their variability due to age and education: a spherical deconvolution tractography study. *Brain Structure and Function*, 221(3), 1751–1766. https://doi.org/10.1007/s00429-015-1001-3
---

## Contact

Irene Bellin — [GitHub](https://github.com/irenebellin/ECoG)  
PhD candidate, University of Padova / Donders Centre for Cognition, Radboud University
