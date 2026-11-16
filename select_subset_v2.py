"""
Step 1 (updated): Select 12 diseases + normal (No Finding) cases from
CheXpert labels, filter frontal views, apply official MIMIC split,
and build an image list to download.
"""

import pandas as pd

DATA_DIR = "."

SELECTED_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]

TARGET_PER_CLASS = 3000       # cap per disease (rare ones will just take all available)
TARGET_NORMAL = 6000          # how many "No Finding" (normal) images to add

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

# --- Disease-positive rows (same as before) ---
selected_rows = []
for disease in SELECTED_DISEASES:
    positives = merged[merged[disease] == 1]
    take = positives.sample(n=min(TARGET_PER_CLASS, len(positives)), random_state=42)
    selected_rows.append(take)
    print(f"{disease}: {len(positives)} available, took {len(take)}")

# --- Normal (No Finding) rows — NEW ---
normals = merged[merged["No Finding"] == 1]
normal_take = normals.sample(n=min(TARGET_NORMAL, len(normals)), random_state=42)
selected_rows.append(normal_take)
print(f"Normal (No Finding): {len(normals)} available, took {len(normal_take)}")

subset = pd.concat(selected_rows).drop_duplicates(subset=["dicom_id"])
print("\nTotal unique images selected:", len(subset))

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
print("\nNormal vs has-disease breakdown:")
print("  Normal (No Finding=1):", int(subset["No Finding"].sum()))
print("  Has at least one disease:", int((subset[SELECTED_DISEASES].sum(axis=1) > 0).sum()))