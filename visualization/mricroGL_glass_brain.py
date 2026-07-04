import gl
import os

# =============================================================================
# MRIcroGL script -- glass brain 3D render of disconnectome heatmap
# Run in MRIcroGL: Scripting -> Open -> Ctrl+R
#
# This script cannot be run from the command line -- it must be executed
# from within MRIcroGL's scripting interface (Scripting -> Open -> Ctrl+R).
# The gl module is only available inside MRIcroGL.
#
# Tested with MRIcroGL v1.2 (https://www.nitrc.org/projects/mricrogl)
# =============================================================================

# ---- EDIT THESE PATHS ----
HEATMAP_FILE = r"C:\path\to\heatmaps\grand_average\grand_average_heatmap.nii.gz"
OUT_DIR      = r"C:\path\to\figures"
# ---------------------------

HEMISPHERE   = -1        # -1 = left sagittal, 1 = right sagittal
VMIN         = 0.0002
VMAX         = 0.021
COLORMAP     = "jet"    
MESH_OPACITY = 0.15      # 0=invisible brain, 1=fully opaque

gl.resetdefaults()
gl.loadimage("mni_icbm152_gm_tal_nlin_sym_09a")
gl.overlayloadsmooth(0)
gl.overlayload(HEATMAP_FILE)
gl.backcolor(255, 255, 255)
gl.shadername("Default")
gl.opacity(0, 10)
gl.opacity(1, 100)
gl.colorname(1, COLORMAP)
gl.minmax(1, VMIN, VMAX)
gl.renderquality1to10(10)
gl.shaderquality1to10(10)
gl.windowposition(0, 0, 2400, 2400)
gl.cameradistance(2.0)

if HEMISPHERE == -1:
    gl.viewsagittal(0)
    suffix = "_left"
else:
    gl.viewsagittal(1)
    suffix = "_right"

# View --> here deselect only the one of interest
#gl.viewsagittal(0) #0right, 1left
#gl.viewcoronal(0) #0frontal
#gl.viewaxial(0)

gl.wait(3000)

basename = os.path.basename(HEATMAP_FILE).replace(".nii.gz", "").replace(".nii", "")
os.makedirs(OUT_DIR, exist_ok=True)
gl.savebmp(os.path.join(OUT_DIR, basename + suffix + ".png"))
