import pandas as pd

DATA_DIR = "."

subset = pd.read_csv(f"{DATA_DIR}/subset_labels.csv")
total = len(subset)

SELECTED_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]

print(f"Total images in subset: {total}\n")
print(f"{'Category':30s} {'Positive':>10s} {'Negative':>10s}")
print("-" * 52)

no_finding_pos = int(subset["No Finding"].sum())
print(f"{'Normal (No Finding)':30s} {no_finding_pos:10d} {total - no_finding_pos:10d}")

for d in SELECTED_DISEASES:
    pos = int(subset[d].sum())
    neg = total - pos
    print(f"{d:30s} {pos:10d} {neg:10d}")