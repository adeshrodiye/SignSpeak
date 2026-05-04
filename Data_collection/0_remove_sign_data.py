"""
remove_sign_data.py
===================
Remove one or more specific sign labels from keypoints_dataset.csv.

Usage:
    python remove_sign_data.py
    
Edit the SIGNS_TO_REMOVE list below before running.
"""

import pandas as pd
import os
import shutil
from utils import LABELS

# ── CONFIG — edit this list with signs you want to remove ─────────────────
SIGNS_TO_REMOVE = ["V"]   # e.g. ["9A", "TEACHER", "G"]

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "data", "keypoints_dataset.csv")
BAK_PATH = os.path.join(BASE, "data", "keypoints_dataset_backup.csv")

# ── Load ───────────────────────────────────────────────────────────────────
print(f"\nLoading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, header=None)
total_before = len(df)

print(f"Total rows before: {total_before}")
print("\nCurrent sample counts per label:")
for idx, lbl in enumerate(LABELS):
    count = (df.iloc[:, -1] == idx).sum()
    if count > 0:
        print(f"  [{idx:2d}] {lbl:<12s}  {count} samples")

# ── Resolve indices to remove ──────────────────────────────────────────────
indices_to_remove = []
for sign in SIGNS_TO_REMOVE:
    sign_upper = sign.upper()
    if sign_upper in LABELS:
        idx = LABELS.index(sign_upper)
        indices_to_remove.append(idx)
        print(f"\nWill remove: '{sign_upper}' (label index {idx})")
    else:
        # Try to find by index number directly
        try:
            idx = int(sign)
            indices_to_remove.append(idx)
            print(f"\nWill remove index {idx} ({LABELS[idx] if idx < len(LABELS) else 'unknown'})")
        except ValueError:
            print(f"\nWARNING: '{sign}' not found in LABELS list — skipping.")

if not indices_to_remove:
    print("\nNo valid signs to remove. Exiting.")
    exit()

# ── Backup original CSV ────────────────────────────────────────────────────
shutil.copy(CSV_PATH, BAK_PATH)
print(f"\nBackup saved -> {BAK_PATH}")

# ── Remove rows ────────────────────────────────────────────────────────────
mask = df.iloc[:, -1].isin(indices_to_remove)
removed_count = mask.sum()
df_clean = df[~mask]

# ── Save ───────────────────────────────────────────────────────────────────
df_clean.to_csv(CSV_PATH, header=False, index=False)

print(f"\nRows removed : {removed_count}")
print(f"Rows remaining: {len(df_clean)}")
print(f"\nUpdated sample counts per label:")
for idx, lbl in enumerate(LABELS):
    count = (df_clean.iloc[:, -1] == idx).sum()
    if count > 0:
        print(f"  [{idx:2d}] {lbl:<12s}  {count} samples")

print(f"\nDone! Clean CSV saved -> {CSV_PATH}")
print(f"Original backed up  -> {BAK_PATH}")
print("\nNow fix utils.py if needed, then re-run 1_collect_data.py.")
