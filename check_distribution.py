import pandas as pd

DATA_DIR = "."

subset = pd.read_csv(f"{DATA_DIR}/subset_labels.csv")

SELECTED_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]

print("Total images in current subset:", len(subset))
print("\nPer-disease positive counts:")
for d in SELECTED_DISEASES:
    print(f"  {d}: {int(subset[d].sum())}")

# how many images have NO disease at all in this subset (should be ~0,
# since we only sampled disease-positive rows)
subset["any_disease"] = subset[SELECTED_DISEASES].sum(axis=1) > 0
print("\nImages with at least one disease:", int(subset["any_disease"].sum()))
print("Images with NO disease label set (normal-like in this subset):",
      int((~subset["any_disease"]).sum()))