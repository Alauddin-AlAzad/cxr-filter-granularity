"""
Rule-based finding extraction from generated report text, with negation handling.
This is an approximation of the CheXpert labeler, not a replacement - state that
clearly in the paper. It is enough to compare two models on the same test set.
"""

import re

import numpy as np
import pandas as pd

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]

PATTERNS = {
    "Atelectasis": ["atelecta", "collapse of the", "volume loss"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "heart is enlarged",
                     "cardiac enlargement", "enlargement of the cardiac silhouette"],
    "Consolidation": ["consolidat"],
    "Edema": ["edema", "vascular congestion", "pulmonary congestion"],
    "Enlarged Cardiomediastinum": ["mediastinal widening", "widened mediastinum",
                                   "enlarged cardiomediastinal", "cardiomediastinal silhouette is enlarged"],
    "Fracture": ["fracture"],
    "Lung Lesion": ["nodule", "mass", "lesion", "cavitary"],
    "Lung Opacity": ["opacit", "opacification", "infiltrat", "densit"],
    "Pleural Effusion": ["pleural effusion", "effusion", "pleural fluid"],
    "Pleural Other": ["pleural thickening", "pleural scarring", "fibrosis",
                      "pleural plaque", "empyema"],
    "Pneumonia": ["pneumonia", "infectio"],
    "Pneumothorax": ["pneumothorax", "pneumothoraces"],
}

NEGATIONS = [
    "no ", "no evidence of", "without", "free of", "negative for", "absence of",
    "not ", "rule out", "ruled out", "unremarkable", "clear", "resolved",
]

_SENT = re.compile(r"(?<=[.;])\s+|\.\s*$")


def _is_negated(sentence, hit_pos):
    window = sentence[max(0, hit_pos - 60):hit_pos]
    return any(n in window for n in NEGATIONS)


def extract_findings(text):
    text = str(text).lower()
    out = {c: 0 for c in LABELS}
    for sent in _SENT.split(text):
        if not sent:
            continue
        for label, keys in PATTERNS.items():
            for k in keys:
                p = sent.find(k)
                if p >= 0 and not _is_negated(sent, p):
                    out[label] = 1
                    break
    return out


def clinical_efficacy(hyp_texts, true_labels):
    pred = pd.DataFrame([extract_findings(t) for t in hyp_texts])[LABELS].values
    true = np.asarray(true_labels)[:, : len(LABELS)]

    rows = []
    for i, c in enumerate(LABELS):
        tp = int(((pred[:, i] == 1) & (true[:, i] == 1)).sum())
        fp = int(((pred[:, i] == 1) & (true[:, i] == 0)).sum())
        fn = int(((pred[:, i] == 0) & (true[:, i] == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"finding": c, "support": int(true[:, i].sum()),
                     "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})

    df = pd.DataFrame(rows)
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    summary = {"macro_f1": round(df["f1"].mean(), 4),
               "micro_precision": round(micro_p, 4),
               "micro_recall": round(micro_r, 4),
               "micro_f1": round(micro_f, 4)}
    return df, summary