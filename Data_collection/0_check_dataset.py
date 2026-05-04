# import pandas as pd
# from utils import LABELS

# df = pd.read_csv("data/keypoints_dataset.csv", header=None)
# print("Sample counts per label:")
# for idx, lbl in enumerate(LABELS):
#     count = (df.iloc[:, -1] == idx).sum()
#     status = "✓" if count > 0 else "✗ MISSING"
#     print(f"  [{idx:2d}] {lbl:<12s}  {count} samples  {status}")

import pandas as pd
from utils import LABELS

df = pd.read_csv("data/keypoints_dataset.csv", header=None)
print(f"Total samples: {len(df)}")
print(f"Total classes in utils.py: {len(LABELS)}")
print(f"Unique indices in CSV: {sorted(df.iloc[:, -1].unique())}")
print(f"Expected indices: {list(range(len(LABELS)))}")

# Find any missing
missing = [i for i in range(len(LABELS)) if (df.iloc[:, -1] == i).sum() == 0]
if missing:
    print(f"\nWARNING — these labels have 0 samples: {[LABELS[i] for i in missing]}")
else:
    print("\nAll labels have samples. Safe to train!")