import pandas as pd

DATA_DIR = "."

SELECTED_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]

chexpert = pd.read_csv(f"{DATA_DIR}/mimic-cxr-2.0.0-chexpert.csv")
metadata = pd.read_csv(f"{DATA_DIR}/mimic-cxr-2.0.0-metadata.csv")
split = pd.read_csv(f"{DATA_DIR}/mimic-cxr-2.0.0-split.csv")

frontal = metadata[metadata["ViewPosition"].isin(["AP", "PA"])]
merged = frontal.merge(chexpert, on=["subject_id", "study_id"], how="inner")
merged = merged.merge(split, on=["dicom_id", "subject_id", "study_id"], how="inner")

print("Total frontal images (all studies, after merge):", len(merged))
print()

for d in SELECTED_DISEASES + ["No Finding"]:
    merged[d] = merged[d].fillna(0)
    merged[d] = merged[d].replace(-1, 0)

print("Full availability (entire dataset, no cap applied):")
total_disease = 0
for disease in SELECTED_DISEASES:
    count = int((merged[disease] == 1).sum())
    total_disease += count
    print(f"  {disease}: {count}")

normal_count = int((merged["No Finding"] == 1).sum())
print(f"  Normal (No Finding): {normal_count}")

print("\nSum across 12 diseases (note: one image can have multiple diseases,")
print("so this sum double-counts):", total_disease)

unique_with_any_disease = int((merged[SELECTED_DISEASES].sum(axis=1) > 0).sum())
print("Unique images with at least 1 of the 12 diseases:", unique_with_any_disease)
print("Unique normal images:", normal_count)
print("Grand total available (disease + normal, unique):",
      unique_with_any_disease + normal_count)