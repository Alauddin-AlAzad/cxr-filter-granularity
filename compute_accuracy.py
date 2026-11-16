"""
Computes classification accuracy on the test split, alongside the baseline a
model would reach by answering "no finding" to everything. Prints both so the
difference is visible.

Needs: prompt_probs.csv, prompt_thresholds.json, available_labels.csv
"""

import json
import os

import numpy as np
import pandas as pd

from cxr_data_224 import LABEL_COLS, load_frames

DATA_DIR = "."

probs = pd.read_csv(os.path.join(DATA_DIR, "prompt_probs.csv"))
thresholds = json.load(open(os.path.join(DATA_DIR, "prompt_thresholds.json")))

_, _, test_df = load_frames(DATA_DIR)
t = test_df[["dicom_id"] + LABEL_COLS].merge(probs, on="dicom_id", how="inner")
print("Test images:", len(t))

y = t[LABEL_COLS].fillna(0).replace(-1, 0).values.astype(int)
p = t[[f"p_{c}" for c in LABEL_COLS]].values

pred = np.zeros_like(y)
for i, c in enumerate(LABEL_COLS):
    th = thresholds.get(c, 0.5)
    th = 0.5 if th > 1.0 else th
    pred[:, i] = (p[:, i] >= th).astype(int)

per_label = (pred == y).mean(axis=0)
overall = (pred == y).mean()
exact = (pred == y).all(axis=1).mean()

always_negative = (y == 0).mean()
always_neg_exact = (y.sum(axis=1) == 0).mean()

print("\nPer-label accuracy:")
for c, a in zip(LABEL_COLS, per_label):
    print(f"  {c:30s} {a * 100:5.1f}%")

print(f"\nOverall accuracy (all 13 labels)      : {overall * 100:.1f}%")
print(f"Exact match (all 13 correct at once)  : {exact * 100:.1f}%")
print(f"\nBaseline - always answer 'no finding' : {always_negative * 100:.1f}%")
print(f"Baseline exact match                  : {always_neg_exact * 100:.1f}%")
print(f"\nGain over the do-nothing baseline     : {(overall - always_negative) * 100:+.1f} points")
print("\nMean AUC on the same split is the metric reported in the paper.")