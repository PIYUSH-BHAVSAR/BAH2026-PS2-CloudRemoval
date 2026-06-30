# download_from_drive.py
# Downloads all 9 GeoTIFFs from public Google Drive links using gdown.
# No login needed — files are publicly shared.

import os
import subprocess
import sys

# Install gdown if not present
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
    import gdown

os.makedirs('ps2_cloud/data/raw/sentinel1', exist_ok=True)
os.makedirs('ps2_cloud/data/raw/sentinel2', exist_ok=True)

# ── File ID map ──────────────────────────────────────────────────────────
# Format: 'filename': 'google_drive_file_id'
# File ID is the part after /d/ in the share link
# e.g. https://drive.google.com/file/d/1OcAd_KhuEiZYSbYiXJ619unN58p6hJ6x/view
#                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this part

FILES = {
    # Bengaluru — file ID from your share link
    'bengaluru_S2_cloudy.tif': '1OcAd_KhuEiZYSbYiXJ619unN58p6hJ6x',

    # Remaining 8 — from folder: 1-MmELXDpLpHysaauFhBYDGfFkW-Lqstf
    # gdown will list and download all files from the folder
}

FOLDER_ID = '1-MmELXDpLpHysaauFhBYDGfFkW-Lqstf'

def get_output_path(filename):
    if 'S1_sar' in filename:
        return f'ps2_cloud/data/raw/sentinel1/{filename}'
    return f'ps2_cloud/data/raw/sentinel2/{filename}'

# ── Download bengaluru_S2_cloudy.tif individually ────────────────────────
print("Downloading bengaluru_S2_cloudy.tif ...")
out = get_output_path('bengaluru_S2_cloudy.tif')
if os.path.exists(out):
    print(f"  Already exists: {out}")
else:
    gdown.download(
        f"https://drive.google.com/uc?id=1OcAd_KhuEiZYSbYiXJ619unN58p6hJ6x",
        out, quiet=False
    )
    if os.path.exists(out):
        print(f"  Saved: {out}")

# ── Download all 8 remaining files from folder ───────────────────────────
print("\nDownloading remaining 8 files from folder ...")
print("Files will download to current directory first, then be moved.\n")

# Download folder contents to a temp dir
tmp_dir = 'ps2_cloud/data/raw/drive_tmp'
os.makedirs(tmp_dir, exist_ok=True)

gdown.download_folder(
    f"https://drive.google.com/drive/folders/{FOLDER_ID}",
    output=tmp_dir,
    quiet=False,
    use_cookies=False,
)

# ── Move files to correct folders ────────────────────────────────────────
print("\nMoving files to correct folders...")
import shutil

moved = 0
for fname in os.listdir(tmp_dir):
    if fname.endswith('.tif'):
        src = os.path.join(tmp_dir, fname)
        dst = get_output_path(fname)
        if os.path.exists(dst):
            print(f"  Already exists: {dst}")
        else:
            shutil.move(src, dst)
            mb = os.path.getsize(dst) / 1024 / 1024
            print(f"  Moved: {fname} ({mb:.1f} MB) → {dst}")
            moved += 1

# Clean up tmp dir
shutil.rmtree(tmp_dir, ignore_errors=True)

# ── Final check ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
expected = [
    'ps2_cloud/data/raw/sentinel2/bengaluru_S2_cloudy.tif',
    'ps2_cloud/data/raw/sentinel2/bengaluru_S2_clear.tif',
    'ps2_cloud/data/raw/sentinel2/punjab_S2_cloudy.tif',
    'ps2_cloud/data/raw/sentinel2/punjab_S2_clear.tif',
    'ps2_cloud/data/raw/sentinel2/meghalaya_S2_cloudy.tif',
    'ps2_cloud/data/raw/sentinel2/meghalaya_S2_clear.tif',
    'ps2_cloud/data/raw/sentinel1/bengaluru_S1_sar.tif',
    'ps2_cloud/data/raw/sentinel1/punjab_S1_sar.tif',
    'ps2_cloud/data/raw/sentinel1/meghalaya_S1_sar.tif',
]

all_ok = True
for f in expected:
    if os.path.exists(f):
        mb = os.path.getsize(f) / 1024 / 1024
        print(f"  ✓  {f}  ({mb:.1f} MB)")
    else:
        print(f"  ✗  MISSING: {f}")
        all_ok = False

print(f"{'='*50}")
if all_ok:
    print("\nAll 9 files present. Phase 1 data collection COMPLETE.")
    print("Next: python preprocess.py")
else:
    print("\nSome files missing — check above.")
