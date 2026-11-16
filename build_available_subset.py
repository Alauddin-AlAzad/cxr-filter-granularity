"""
Builds a labels CSV limited to images that are on disk AND fully downloaded.
Re-run this any time; as more images finish downloading the subset grows.
"""

import os

import pandas as pd

DATA_DIR = "."
IMAGE_ROOT = "images"
MIN_BYTES = 20000

SELECTED_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]


def is_complete_jpeg(path):
    try:
        if os.path.getsize(path) < MIN_BYTES:
            return False
        with open(path, "rb") as f:
            f.seek(-2, os.SEEK_END)
            return f.read(2) == b"\xff\xd9"
    except OSError:
        return False


labels = pd.read_csv(f"{DATA_DIR}/subset_labels.csv")
print("Planned images in full subset:", len(labels))

labels["local_path"] = labels["image_path"].apply(
    lambda p: os.path.join(DATA_DIR, IMAGE_ROOT, p)
)

on_disk = labels["local_path"].apply(os.path.isfile)
print("Files present on disk:", int(on_disk.sum()))

labels["usable"] = False
labels.loc[on_disk, "usable"] = labels.loc[on_disk, "local_path"].apply(is_complete_jpeg)

incomplete = int(on_disk.sum() - labels["usable"].sum())
print("Incomplete / unreadable (skipped):", incomplete)

available = labels[labels["usable"]].copy()
print("Usable images:", len(available))
print(f"Coverage: {len(available) / len(labels) * 100:.1f}%")

available.drop(columns=["usable"]).to_csv(
    f"{DATA_DIR}/available_labels.csv", index=False
)

print("\nPer-class positives in what's available now:")
print(f"  {'Normal (No Finding)':30s}: {int(available['No Finding'].sum())}")
for d in SELECTED_DISEASES:
    print(f"  {d:30s}: {int(available[d].sum())}")

print("\nSplit counts:")
print(available["split"].value_counts())
print("\nSaved: available_labels.csv")