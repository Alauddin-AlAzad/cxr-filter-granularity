"""
Step 1: Select 12 diseases from CheXpert labels, filter frontal views,
apply official MIMIC split, and build an image list to download.

Run this on your own PC (not here) after downloading:
  mimic-cxr-2.0.0-chexpert.csv
  mimic-cxr-2.0.0-metadata.csv
  mimic-cxr-2.0.0-split.csv
  IMAGE_FILENAMES
into the same folder as this script.
"""

import pandas as pd

DATA_DIR = "."  # change if your CSVs are in a different folder

# 12 diseases (from the standard 14 CheXpert labels; dropping
# "No Finding" and "Support Devices" since those aren't diagnoses)
SELECTED_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]

TARGET_PER_CLASS = 3000  # roughly how many positive cases per disease to keep

# --- Load CSVs ---
chexpert = pd.read_csv(f"{DATA_DIR}/mimic-cxr-2.0.0-chexpert.csv")
metadata = pd.read_csv(f"{DATA_DIR}/mimic-cxr-2.0.0-metadata.csv")
split = pd.read_csv(f"{DATA_DIR}/mimic-cxr-2.0.0-split.csv")

# --- Keep only frontal views (AP or PA) ---
frontal = metadata[metadata["ViewPosition"].isin(["AP", "PA"])]
print("Frontal-view rows:", len(frontal))

# --- Merge labels with frontal metadata and split info ---
merged = frontal.merge(chexpert, on=["subject_id", "study_id"], how="inner")
merged = merged.merge(split, on=["dicom_id", "subject_id", "study_id"], how="inner")
print("After merging labels+split:", len(merged))

# --- Treat uncertain (-1) as negative for simplicity (documented choice) ---
for d in SELECTED_DISEASES:
    merged[d] = merged[d].fillna(0)
    merged[d] = merged[d].replace(-1, 0)

# --- Sample a balanced subset per disease ---
selected_rows = []
for disease in SELECTED_DISEASES:
    positives = merged[merged[disease] == 1]
    take = positives.sample(n=min(TARGET_PER_CLASS, len(positives)), random_state=42)
    selected_rows.append(take)
    print(f"{disease}: {len(positives)} available, took {len(take)}")

subset = pd.concat(selected_rows).drop_duplicates(subset=["dicom_id"])
print("\nTotal unique images selected:", len(subset))

# --- Build the relative image path (matches MIMIC-CXR-JPG folder structure) ---
subset["image_path"] = (
    "files/p" + subset["subject_id"].astype(str).str[:2] + "/"
    "p" + subset["subject_id"].astype(str) + "/"
    "s" + subset["study_id"].astype(str) + "/"
    + subset["dicom_id"] + ".jpg"
)

# --- Save the list for download + a labels file for training ---
subset[["image_path"]].to_csv(f"{DATA_DIR}/images_to_download.txt", index=False, header=False)
subset[["dicom_id", "subject_id", "study_id", "image_path", "split"] + SELECTED_DISEASES].to_csv(
    f"{DATA_DIR}/subset_labels.csv", index=False
)

print("\nSaved: images_to_download.txt (for wget -i)")
print("Saved: subset_labels.csv (for classifier training)")
print("\nSplit counts:")
print(subset["split"].value_counts())