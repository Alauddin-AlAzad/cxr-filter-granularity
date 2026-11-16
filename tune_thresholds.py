"""
Re-tunes the prompting thresholds from the probabilities already saved by
predict_findings.py. Runs in seconds; no GPU pass needed.

A finding is only prompted if the classifier can assert it with at least
MIN_PRECISION precision on the validation split. Among the thresholds that
clear that bar, the one maximising F0.5 is chosen, so precision is weighted
more heavily than recall: a wrong prompt actively misleads the decoder,
whereas a missing prompt only leaves it where it was before.

Output: prompt_thresholds.json
"""

import json
import os

import numpy as np
import pandas as pd

from cxr_data_224 import LABEL_COLS, load_frames

DATA_DIR = "."
MIN_PRECISION = 0.50
BETA = 0.5
NEVER = 1.01


def main():
    probs = pd.read_csv(os.path.join(DATA_DIR, "prompt_probs_calibrated.csv"))
    _, val_df, _ = load_frames(DATA_DIR)

    v = val_df[["dicom_id"] + LABEL_COLS].merge(probs, on="dicom_id", how="inner")
    print("Validation images used:", len(v))

    y = v[LABEL_COLS].fillna(0).replace(-1, 0).values.astype(int)
    p = v[[f"p_{c}" for c in LABEL_COLS]].values

    b2 = BETA ** 2
    thresholds, rows = {}, []

    for i, c in enumerate(LABEL_COLS):
        best = None
        for t in np.arange(0.05, 0.99, 0.025):
            pred = (p[:, i] >= t).astype(int)
            tp = int(((pred == 1) & (y[:, i] == 1)).sum())
            fp = int(((pred == 1) & (y[:, i] == 0)).sum())
            fn = int(((pred == 0) & (y[:, i] == 1)).sum())
            if tp + fp == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn) if tp + fn else 0.0
            if prec < MIN_PRECISION:
                continue
            f = ((1 + b2) * prec * rec / (b2 * prec + rec)) if (prec + rec) else 0.0
            if best is None or f > best[0]:
                best = (f, float(t), prec, rec, int(pred.sum()))

        if best is None:
            thresholds[c] = NEVER
            rows.append({"finding": c, "threshold": "never", "precision": "-",
                         "recall": "-", "fires": 0})
        else:
            f, t, prec, rec, fires = best
            thresholds[c] = round(t, 3)
            rows.append({"finding": c, "threshold": round(t, 3),
                         "precision": round(prec, 3), "recall": round(rec, 3),
                         "fires": fires})

    print()
    print(pd.DataFrame(rows).to_string(index=False))

    pred = np.zeros_like(y)
    for i, c in enumerate(LABEL_COLS):
        pred[:, i] = (p[:, i] >= thresholds[c]).astype(int)
    per_image = pred[:, 1:].sum(1)
    true_per_image = y[:, 1:].sum(1)

    print(f"\nFindings prompted per image : mean {per_image.mean():.2f}  median {np.median(per_image):.0f}")
    print(f"True findings per image     : mean {true_per_image.mean():.2f}  median {np.median(true_per_image):.0f}")
    print(f"Images with an empty prompt : {(per_image == 0).mean() * 100:.1f}%")

    never = [c for c in LABEL_COLS if thresholds[c] == NEVER]
    if never:
        print("\nNever prompted (classifier cannot reach "
              f"{MIN_PRECISION:.0%} precision): {', '.join(never)}")

    json.dump(thresholds, open(os.path.join(DATA_DIR, "prompt_thresholds_calibrated.json"), "w"), indent=2)
    print("\nSaved: prompt_thresholds_calibrated.json")


if __name__ == "__main__":
    main()