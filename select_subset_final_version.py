"""
FINAL (locked) selection script — cap=10000 per disease, normal=15000.
This is the final subset for the whole project. No more changes after this.
"""

import pandas as pd

DATA_DIR = "."

SELECTED_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]

TARGET_PER_CLASS = 10000
TARGET_NORMAL = 15000

chexpert = pd.read_csv(f"{DATA_DIR}/mimic-cxr-2.0.0-chexpert.csv")
metadata = pd.read_csv(f"{DATA_DIR}/mimic-cxr-2.0.0-metadata.csv")
split = pd.read_csv(f"{DATA_DIR}/mimic-cxr-2.0.0-split.csv")

frontal = metadata[metadata["ViewPosition"].isin(["AP", "PA"])]
merged = frontal.merge(chexpert, on=["subject_id", "study_id"], how="inner")
merged = merged.merge(split, on=["dicom_id", "subject_id", "study_id"], how="inner")
print("After merging labels+split:", len(merged))

for d in SELECTED_DISEASES + ["No Finding"]:
    merged[d] = merged[d].fillna(0)
    merged[d] = merged[d].replace(-1, 0)

selected_rows = []
for disease in SELECTED_DISEASES:
    positives = merged[merged[disease] == 1]
    take = positives.sample(n=min(TARGET_PER_CLASS, len(positives)), random_state=42)
    selected_rows.append(take)
    print(f"{disease}: {len(positives)} available, took {len(take)}")

normals = merged[merged["No Finding"] == 1]
normal_take = normals.sample(n=min(TARGET_NORMAL, len(normals)), random_state=42)
selected_rows.append(normal_take)
print(f"Normal (No Finding): {len(normals)} available, took {len(normal_take)}")

subset = pd.concat(selected_rows).drop_duplicates(subset=["dicom_id"])
print("\nFINAL total unique images selected:", len(subset))

subset["image_path"] = (
    "files/p" + subset["subject_id"].astype(str).str[:2] + "/"
    "p" + subset["subject_id"].astype(str) + "/"
    "s" + subset["study_id"].astype(str) + "/"
    + subset["dicom_id"] + ".jpg"
)

subset[["image_path"]].to_csv(f"{DATA_DIR}/images_to_download.txt", index=False, header=False)
subset[["dicom_id", "subject_id", "study_id", "image_path", "split", "No Finding"] + SELECTED_DISEASES].to_csv(
    f"{DATA_DIR}/subset_labels.csv", index=False
)

print("\nSaved: images_to_download.txt (for wget -i)")
print("Saved: subset_labels.csv (for classifier training)")
print("\nSplit counts:")
print(subset["split"].value_counts())

est_gb = len(subset) * 700 / (1024 * 1024)
print(f"\nEstimated download size: ~{est_gb:.1f} GB")
print("\n*** FINAL LOCKED SUBSET. Proceed to download. ***")