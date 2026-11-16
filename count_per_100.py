"""
Turns the test-set metrics into plain counts, out of 100.

F1 alone cannot answer "how many of 100 did we get right", because the same F1
comes from very different precision and recall. This prints both, for the
classifier and for the generated reports, plus a per-disease breakdown.

    python count_per_100.py
    python count_per_100.py --tag uncal224

Reads prompt_probs.csv, prompt_thresholds.json and
checkpoints_gen_<tag>/test_generated.json. No GPU needed.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from cxr_data import LABEL_COLS
from gen_data import load_splits
from clinical_eval import extract_findings, LABELS

UNDETECTABLE = ["Enlarged Cardiomediastinum"]
RUNS = {
    "phrase224": ("uncalibrated", "phrase"),
    "uncal224": ("uncalibrated", "filtered"),
    "oracle_phrase": ("oracle", "phrase"),
    "none_phrase": ("none", "phrase"),
    "unfilt224": ("uncalibrated", "unfiltered"),
}


def counts(pred, true):
    """pred and true are 0/1 arrays of the same shape."""
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    return tp, fp, fn, tn


def rates(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def report_block(title, pred, true, labels, n_img):
    tp, fp, fn, tn = counts(pred, true)
    p, r, f = rates(tp, fp, fn)
    print("\n" + title)
    print("  images: %d   findings checked per image: %d" % (n_img, len(labels)))
    print("  micro precision %.3f   recall %.3f   F1 %.3f" % (p, r, f))
    print("  out of 100 findings it calls, %.0f are really there" % (100 * p))
    print("  out of 100 findings really there, it calls %.0f" % (100 * r))

    any_true = true.any(axis=1)
    any_pred = pred.any(axis=1)
    agree = int((any_true == any_pred).sum())
    print("  'anything wrong at all?' answered correctly on %d of 100 images"
          % round(100 * agree / n_img))

    rows = []
    for j, lab in enumerate(labels):
        a, b, c, _ = counts(pred[:, j], true[:, j])
        pp, rr, ff = rates(a, b, c)
        rows.append({"finding": lab, "in test set": a + c, "called": a + b,
                     "correct": a, "precision": round(pp, 3),
                     "recall": round(rr, 3), "F1": round(ff, 3)})
    df = pd.DataFrame(rows).sort_values("in test set", ascending=False)
    print(df.to_string(index=False))
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "any_finding_agreement": round(agree / n_img, 4),
            "per_finding": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="phrase224")
    args = ap.parse_args()
    if args.tag not in RUNS:
        print("tag must be one of %s" % ", ".join(RUNS))
        return
    prompts, reports = RUNS[args.tag]
    out = {"tag": args.tag}

    _, _, te = load_splits(".", prompts, reports)
    labels = [l for l in LABELS if l not in UNDETECTABLE]

    # the classifier, at the thresholds the prompts use
    probs_path, th_path = "prompt_probs.csv", "prompt_thresholds.json"
    if os.path.isfile(probs_path) and os.path.isfile(th_path):
        probs = pd.read_csv(probs_path)
        th = json.load(open(th_path))
        m = te[["dicom_id"] + labels].merge(probs, on="dicom_id", how="inner")
        true = m[labels].fillna(0).replace(-1, 0).values.astype(int).astype(bool)
        pred = np.zeros_like(true)
        for j, lab in enumerate(labels):
            col = "p_" + lab
            t = th.get(lab, 0.5)
            t = float(t) if isinstance(t, (int, float)) and 0 < t <= 1 else 0.5
            if col in m:
                pred[:, j] = m[col].fillna(0).values >= t
        out["classifier"] = report_block("CLASSIFIER (image -> 12 findings)",
                                         pred, true, labels, len(m))
    else:
        print("no %s / %s, skipping the classifier" % (probs_path, th_path))

    # the generated reports
    gen = os.path.join("checkpoints_gen_" + args.tag, "test_generated.json")
    if os.path.isfile(gen):
        data = json.load(open(gen))
        true_all = te[labels].fillna(0).replace(-1, 0).values.astype(int).astype(bool)
        n = min(len(data), len(true_all))
        true = true_all[:n]
        pred = np.zeros_like(true)
        for i, d in enumerate(data[:n]):
            f = extract_findings(d["hyp"])
            for j, lab in enumerate(labels):
                pred[i, j] = bool(f.get(lab, 0))
        out["generator"] = report_block("GENERATED REPORTS (%s)" % args.tag,
                                        pred, true, labels, n)

        # what the reference reports themselves say, as a ceiling
        rp = np.zeros_like(true)
        for i, d in enumerate(data[:n]):
            f = extract_findings(d["ref"])
            for j, lab in enumerate(labels):
                rp[i, j] = bool(f.get(lab, 0))
        out["reference_ceiling"] = report_block(
            "REFERENCE REPORTS, same extractor (the ceiling this can reach)",
            rp, true, labels, n)
    else:
        print("\nno %s" % gen)

    with open("count_per_100_%s.json" % args.tag, "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved count_per_100_%s.json" % args.tag)


if __name__ == "__main__":
    main()