"""
Runs the trained classifier over every downloaded image and stores the predicted
probability of each finding, plus a per-class decision threshold tuned on the
validation split.

Input : checkpoints/best.pt, available_labels.csv
Output: prompt_probs.csv, prompt_thresholds.json
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import densenet121

from cxr_data_224 import LABEL_COLS, CXRDataset, eval_tf, load_frames

DATA_DIR = "."
CKPT = "checkpoints_320/best.pt"
BATCH_SIZE = 64
NUM_WORKERS = 4


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABEL_COLS))
    ck = torch.load(os.path.join(DATA_DIR, CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    print(f"Loaded classifier (val mean AUC {ck.get('auc', float('nan')):.4f})")

    train_df, val_df, test_df = load_frames(DATA_DIR)
    full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full = full.drop_duplicates(subset=["dicom_id"]).reset_index(drop=True)
    print("Images to score:", len(full))

    loader = DataLoader(CXRDataset(full, eval_tf), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    probs = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            if i % 100 == 0:
                print(f"  {i * BATCH_SIZE}/{len(full)}")

    probs = np.concatenate(probs)[: len(full)]

    out = pd.DataFrame(probs, columns=[f"p_{c}" for c in LABEL_COLS])
    out.insert(0, "dicom_id", full["dicom_id"].values)
    out.to_csv(os.path.join(DATA_DIR, "prompt_probs.csv"), index=False)
    print("Saved: prompt_probs.csv")

    val_ids = set(val_df["dicom_id"])
    mask = full["dicom_id"].isin(val_ids).values
    yv = full.loc[mask, LABEL_COLS].fillna(0).replace(-1, 0).values.astype(int)
    pv = probs[mask]

    thresholds = {}
    print("\nPer-class threshold tuned on validation (max F1):")
    for i, c in enumerate(LABEL_COLS):
        best_t, best_f = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.05):
            pred = (pv[:, i] >= t).astype(int)
            tp = int(((pred == 1) & (yv[:, i] == 1)).sum())
            fp = int(((pred == 1) & (yv[:, i] == 0)).sum())
            fn = int(((pred == 0) & (yv[:, i] == 1)).sum())
            f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f > best_f:
                best_f, best_t = f, float(t)
        thresholds[c] = round(best_t, 2)
        print(f"  {c:30s} t={best_t:.2f}  F1={best_f:.3f}")

    json.dump(thresholds, open(os.path.join(DATA_DIR, "prompt_thresholds.json"), "w"), indent=2)
    print("\nSaved: prompt_thresholds.json")


if __name__ == "__main__":
    main()