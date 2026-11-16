import pandas as pd

DATA_DIR = "."

subset = pd.read_csv(f"{DATA_DIR}/subset_labels.csv")

SELECTED_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]

print("Final per-category counts in your saved subset:\n")
print(f"  {'Normal (No Finding)':30s}: {int(subset['No Finding'].sum())}")
for d in SELECTED_DISEASES:
    print(f"  {d:30s}: {int(subset[d].sum())}")

print("\nTotal rows (unique images):", len(subset))