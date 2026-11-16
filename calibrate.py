"""
Calibration analysis for the 13-label classifier.

A probability of 0.90 should mean the finding is present about 90 times in 100.
A model trained with pos_weight is shifted upward, not merely over-spread, so
temperature scaling alone cannot fix it: temperature rescales the logits but has
no intercept. Platt scaling (a * logit + b) adds that intercept, and is the
method this script selects in practice.

Every calibrator is fitted on the validation split, the method is chosen by
validation ECE, and only then is the test split touched. Nothing is tuned on
what it reports.

    python calibrate.py

Needs: prompt_probs.csv, available_labels.csv
Writes: calibration.json, calibration_reliability.png, prompt_probs_calibrated.csv
"""

import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cxr_data import LABEL_COLS

DATA_DIR = "."
N_BINS = 15
EPS = 1e-6


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def load():
    labels = pd.read_csv(os.path.join(DATA_DIR, "available_labels.csv"))
    probs = pd.read_csv(os.path.join(DATA_DIR, "prompt_probs.csv"))
    df = labels.merge(probs, on="dicom_id", how="inner")
    out = {}
    for split in ["validate", "test"]:
        d = df[df["split"] == split]
        y = d[LABEL_COLS].fillna(0).replace(-1, 0).values.astype(float)
        p = np.clip(d[["p_" + c for c in LABEL_COLS]].values.astype(float), EPS, 1 - EPS)
        out[split] = (p, y)
    return out, df


def ece(p, y, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        total += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return total


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def macro(p, y):
    e = float(np.mean([ece(p[:, i], y[:, i]) for i in range(p.shape[1])]))
    b = float(np.mean([brier(p[:, i], y[:, i]) for i in range(p.shape[1])]))
    return {"macro_ece": round(e, 4), "macro_brier": round(b, 4), "nll": round(nll(p, y), 4)}


# ---------------------------------------------------------------- calibrators


def fit_platt(logit, y):
    """Fits q = sigmoid(a * logit + b) by Newton steps on the NLL."""
    a, b = 1.0, 0.0
    for _ in range(200):
        z = a * logit + b
        q = sigmoid(z)
        g = q - y
        ga = float(np.mean(g * logit))
        gb = float(np.mean(g))
        w = q * (1 - q) + 1e-9
        haa = float(np.mean(w * logit * logit))
        hab = float(np.mean(w * logit))
        hbb = float(np.mean(w))
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a -= da
        b -= db
        if abs(da) < 1e-8 and abs(db) < 1e-8:
            break
    return a, b


def fit_temperature(logit, y):
    grid = np.arange(0.20, 5.01, 0.01)
    losses = [nll(sigmoid(logit / T), y) for T in grid]
    return float(grid[int(np.argmin(losses))])


def build_calibrators(p, y):
    """Returns {name: (A, B)} so that calibrated = sigmoid(A * logit + B)."""
    logit = np.log(p / (1 - p))
    k = p.shape[1]

    T_global = fit_temperature(logit.ravel(), y.ravel())
    a_g, b_g = fit_platt(logit.ravel(), y.ravel())

    T_class = np.array([fit_temperature(logit[:, i], y[:, i]) for i in range(k)])
    ab = [fit_platt(logit[:, i], y[:, i]) for i in range(k)]
    a_c = np.array([a for a, _ in ab])
    b_c = np.array([b for _, b in ab])

    return {
        "none": (np.ones(k), np.zeros(k)),
        "temperature_global": (np.full(k, 1.0 / T_global), np.zeros(k)),
        "temperature_per_class": (1.0 / T_class, np.zeros(k)),
        "platt_global": (np.full(k, a_g), np.full(k, b_g)),
        "platt_per_class": (a_c, b_c),
    }


def apply(p, AB):
    A, B = AB
    logit = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    return np.clip(sigmoid(A[None, :] * logit + B[None, :]), EPS, 1 - EPS)


# ---------------------------------------------------------------- reporting


def per_class_table(p_before, p_after, y):
    rows = []
    for i, c in enumerate(LABEL_COLS):
        rows.append({"finding": c,
                     "positives": int(y[:, i].sum()),
                     "prevalence": round(float(y[:, i].mean()), 4),
                     "ECE_before": round(ece(p_before[:, i], y[:, i]), 4),
                     "ECE_after": round(ece(p_after[:, i], y[:, i]), 4),
                     "Brier_before": round(brier(p_before[:, i], y[:, i]), 4),
                     "Brier_after": round(brier(p_after[:, i], y[:, i]), 4)})
    return pd.DataFrame(rows)


def reliability_points(p, y, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() < 5:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def plot(before, after, y, name, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), dpi=150)
    for k, (p, title) in enumerate([(before, "uncalibrated"), (after, "after " + name)]):
        x, yy = reliability_points(p.ravel(), y.ravel())
        ax[k].plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="perfect")
        ax[k].plot(x, yy, "o-", color="#1F4E79", lw=1.8, ms=5, label="observed")
        ax[k].set_xlim(0, 1)
        ax[k].set_ylim(0, 1)
        ax[k].set_xlabel("predicted probability")
        ax[k].set_ylabel("observed frequency")
        ax[k].set_title(title, fontsize=11)
        ax[k].grid(alpha=0.25)
        ax[k].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Reliability on the test split (13 labels pooled)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print("\nsaved", path)


def main():
    data, df = load()
    pv, yv = data["validate"]
    pt, yt = data["test"]
    print("validation %d images   test %d images" % (len(pv), len(pt)))

    cal = build_calibrators(pv, yv)

    print("\nfitted on validation, compared on validation")
    print("  %-24s %10s %12s %8s" % ("method", "macro ECE", "macro Brier", "NLL"))
    val_scores = {}
    for name, AB in cal.items():
        m = macro(apply(pv, AB), yv)
        val_scores[name] = m
        print("  %-24s %10.4f %12.4f %8.4f" % (name, m["macro_ece"], m["macro_brier"], m["nll"]))

    chosen = min(val_scores, key=lambda n: val_scores[n]["macro_ece"])
    print("\nchosen by validation ECE: %s" % chosen)

    A, B = cal[chosen]
    print("  per-class parameters (calibrated = sigmoid(a * logit + b)):")
    for c, a, b in zip(LABEL_COLS, A, B):
        print("    %-28s a=%6.3f   b=%7.3f" % (c, a, b))

    pt_cal = apply(pt, cal[chosen])
    print("\ntest, before and after")
    before = macro(pt, yt)
    after = macro(pt_cal, yt)
    print("  %-24s %10s %12s %8s" % ("", "macro ECE", "macro Brier", "NLL"))
    print("  %-24s %10.4f %12.4f %8.4f" % ("uncalibrated", before["macro_ece"],
                                           before["macro_brier"], before["nll"]))
    print("  %-24s %10.4f %12.4f %8.4f" % (chosen, after["macro_ece"],
                                           after["macro_brier"], after["nll"]))

    print("\nper-class on test")
    cmp = per_class_table(pt, pt_cal, yt)
    print(cmp.to_string(index=False))

    plot(pt, pt_cal, yt, chosen, os.path.join(DATA_DIR, "calibration_reliability.png"))

    allp = np.clip(df[["p_" + c for c in LABEL_COLS]].values.astype(float), EPS, 1 - EPS)
    out = pd.DataFrame(apply(allp, cal[chosen]), columns=["p_" + c for c in LABEL_COLS])
    out.insert(0, "dicom_id", df["dicom_id"].values)
    out.to_csv(os.path.join(DATA_DIR, "prompt_probs_calibrated.csv"), index=False)

    with open(os.path.join(DATA_DIR, "calibration.json"), "w") as f:
        json.dump({"n_bins": N_BINS,
                   "chosen": chosen,
                   "validation_all_methods": val_scores,
                   "params": {c: {"a": round(float(a), 4), "b": round(float(b), 4)}
                              for c, a, b in zip(LABEL_COLS, A, B)},
                   "test_before": before,
                   "test_after": after,
                   "per_class_test": cmp.to_dict(orient="records")}, f, indent=1)

    print("\nsaved calibration.json and prompt_probs_calibrated.csv")
    print("next: tune_thresholds.py against prompt_probs_calibrated.csv")


if __name__ == "__main__":
    main()