# extract_liss4.py
# Extracts all 6 LISS-IV zip files into the correct data/raw/liss4/ subfolders.
# Run from D:\projects\BAH2026\

import zipfile
import os
import shutil

# Zip file → target folder mapping
ZIPS = {
    'bengaluru_cloudy_20260611.zip': 'ps2_cloud/data/raw/liss4/bengaluru_cloudy',
    'bengaluru_clear_20260118.zip':  'ps2_cloud/data/raw/liss4/bengaluru_clear',
    'punjab_cloudy_20250704.zip':    'ps2_cloud/data/raw/liss4/punjab_cloudy',
    'punjab_clear_20251125.zip':     'ps2_cloud/data/raw/liss4/punjab_clear',
    'meghalaya_cloudy_20250717.zip': 'ps2_cloud/data/raw/liss4/meghalaya_cloudy',
    'meghalaya_clear_20260125.zip':  'ps2_cloud/data/raw/liss4/meghalaya_clear',
}

for zip_name, target_dir in ZIPS.items():
    zip_path = zip_name  # in current directory

    if not os.path.exists(zip_path):
        print(f"  NOT FOUND: {zip_name} — skipping")
        continue

    os.makedirs(target_dir, exist_ok=True)

    print(f"\nExtracting {zip_name} ...")
    print(f"  → {target_dir}")

    with zipfile.ZipFile(zip_path, 'r') as z:
        members = z.namelist()
        print(f"  Contents ({len(members)} files):")
        for m in members:
            print(f"    {m}")

        # Extract all to a temp location first
        tmp = target_dir + '_tmp'
        z.extractall(tmp)

    # Flatten: Bhoonidhi zips often have a nested subfolder
    # e.g. RS2A_LISS4_MX_.../BAND2.tif  →  we want BAND2.tif in target_dir
    extracted_files = []
    for root, dirs, files in os.walk(tmp):
        for f in files:
            extracted_files.append(os.path.join(root, f))

    print(f"  Moving {len(extracted_files)} files to {target_dir}/")
    for src in extracted_files:
        fname = os.path.basename(src)
        dst   = os.path.join(target_dir, fname)
        shutil.move(src, dst)
        size_mb = os.path.getsize(dst) / 1024 / 1024
        print(f"    {fname}  ({size_mb:.1f} MB)")

    # Clean up temp dir
    shutil.rmtree(tmp, ignore_errors=True)

# ── Final check ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL STRUCTURE CHECK")
print("="*55)

base = 'ps2_cloud/data/raw/liss4'
expected_folders = [
    'bengaluru_cloudy', 'bengaluru_clear',
    'punjab_cloudy',    'punjab_clear',
    'meghalaya_cloudy', 'meghalaya_clear',
]

all_ok = True
for folder in expected_folders:
    path = os.path.join(base, folder)
    if not os.path.exists(path):
        print(f"  ✗  MISSING folder: {path}")
        all_ok = False
        continue

    files = os.listdir(path)
    tifs  = [f for f in files if f.upper().endswith('.TIF')]
    metas = [f for f in files if 'META' in f.upper() or f.upper().endswith('.TXT')]

    status = "✓" if len(tifs) >= 3 else "⚠"
    print(f"  {status}  {folder}/")
    for f in sorted(files):
        size_mb = os.path.getsize(os.path.join(path, f)) / 1024 / 1024
        print(f"       {f}  ({size_mb:.1f} MB)")

    if len(tifs) < 3:
        print(f"       ⚠ Only {len(tifs)} TIF found — expected BAND2, BAND3, BAND4")
        all_ok = False

print("="*55)
if all_ok:
    print("All LISS-IV scenes extracted correctly.")
    print("Next: python preprocess.py")
else:
    print("Some issues found — check above.")
