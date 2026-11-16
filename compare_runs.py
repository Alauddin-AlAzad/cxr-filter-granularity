"""
Compares the finished generator runs with bootstrap confidence intervals.

A single run gives one number with no sense of how much it would move if the
test set had been drawn slightly differently. Resampling the test set with
replacement answers that, and a paired resample answers whether the difference
between two runs is larger than that movement.

    python compare_runs.py

Reads checkpoints_gen_*/test_generated.json, writes comparison.json.
"""

import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd

from cxr_data import LABEL_COLS
from gen_data import load_splits
from clinical_eval import extract_findings, LABELS

DATA_DIR = "."
N_BOOT = 1000
SEED = 0

# folder suffix -> (prompts, reports) so the right labels are loaded
RUNS = {
    "main224": ("calibrated", "filtered"),
    "uncal224": ("uncalibrated", "filtered"),
    "oracle224": ("oracle", "filtered"),
    "none224": ("none", "filtered"),
    "unfilt224": ("uncalibrated", "unfiltered"),
    "phrase224": ("uncalibrated", "phrase"),
    "oracle_phrase": ("oracle", "phrase"),
    "none_phrase": ("none", "phrase"),
    "cal_phrase": ("calibrated", "phrase"),
    "phrase224_s2": ("uncalibrated", "phrase"),
    "uncal224_s2": ("uncalibrated", "filtered"),
    "calibrated_filtered": ("calibrated", "filtered"),
}

# findings the rule-based extractor cannot detect at all, measured on the
# reference reports; reported separately rather than counted as zero
UNDETECTABLE = ["Enlarged Cardiomediastinum"]


def ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu4_parts(h, r):
    """Per-pair counts, so a bootstrap can re-pool them cheaply."""
    out = {"hl": len(h), "rl": len(r)}
    for n in range(1, 5):
        hn, rn = ngrams(h, n), ngrams(r, n)
        out["num%d" % n] = sum(min(c, rn[g]) for g, c in hn.items())
        out["den%d" % n] = sum(hn.values())
    return out


def bleu4_from_parts(parts, idx):
    hl = sum(parts[i]["hl"] for i in idx)
    rl = sum(parts[i]["rl"] for i in idx)
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    logs = []
    for n in range(1, 5):
        num = sum(parts[i]["num%d" % n] for i in idx)
        den = sum(parts[i]["den%d" % n] for i in idx)
        p = num / den if den else 0.0
        if p <= 0:
            return 0.0
        logs.append(math.log(p))
    return bp * math.exp(sum(logs) / 4)


def rouge_l_pair(h, r):
    if not h or not r:
        return 0.0
    m, n = len(h), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if h[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[m][n]
    p, rc = l / m, l / n
    return 0.0 if p + rc == 0 else 2 * p * rc / (p + rc)


def ce_parts(hyps, true, labels):
    """Per-image tp/fp/fn vectors over the measurable findings."""
    tp, fp, fn = [], [], []
    for h, y in zip(hyps, true):
        pred = extract_findings(h)
        a = b = c = 0
        for j, lab in enumerate(labels):
            p = 1 if pred.get(lab, 0) else 0
            t = int(y[j])
            if p and t:
                a += 1
            elif p and not t:
                b += 1
            elif t and not p:
                c += 1
        tp.append(a)
        fp.append(b)
        fn.append(c)
    return np.array(tp), np.array(fp), np.array(fn)


def micro_f1(tp, fp, fn, idx):
    a, b, c = tp[idx].sum(), fp[idx].sum(), fn[idx].sum()
    p = a / (a + b) if a + b else 0.0
    r = a / (a + c) if a + c else 0.0
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def load_run(tag):
    path = os.path.join("checkpoints_gen_" + tag, "test_generated.json")
    if not os.path.isfile(path):
        return None
    data = json.load(open(path))
    prompts, reports = RUNS[tag]
    _, _, te = load_splits(DATA_DIR, prompts, reports)
    labels = [l for l in LABELS if l not in UNDETECTABLE]
    true = te[labels].fillna(0).replace(-1, 0).values.astype(int)
    n = min(len(data), len(true))
    hyps = [d["hyp"] for d in data[:n]]
    refs = [d["ref"] for d in data[:n]]
    return hyps, refs, true[:n], labels


def summarise(tag, rng):
    loaded = load_run(tag)
    if loaded is None:
        return None
    hyps, refs, true, labels = loaded
    h = [x.lower().split() for x in hyps]
    r = [x.lower().split() for x in refs]

    parts = [bleu4_parts(a, b) for a, b in zip(h, r)]
    rl = np.array([rouge_l_pair(a, b) for a, b in zip(h, r)])
    tp, fp, fn = ce_parts(hyps, true, labels)

    n = len(h)
    full = np.arange(n)
    point = {"BLEU-4": bleu4_from_parts(parts, full),
             "ROUGE-L": float(rl.mean()),
             "CE_micro_F1": micro_f1(tp, fp, fn, full)}

    boots = {k: [] for k in point}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots["BLEU-4"].append(bleu4_from_parts(parts, idx))
        boots["ROUGE-L"].append(float(rl[idx].mean()))
        boots["CE_micro_F1"].append(micro_f1(tp, fp, fn, idx))

    out = {"n": n}
    for k, v in point.items():
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        out[k] = {"value": round(v, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
    return out, (parts, rl, tp, fp, fn)


def paired(a, b, rng, name_a, name_b):
    """Resamples the same image indices for both runs, so the pairing holds.

    Only valid when both runs were evaluated on the same images in the same
    order. A filtered and an unfiltered run are not, so that case is refused
    rather than silently compared.
    """
    pa, ra, tpa, fpa, fna = a
    pb, rb, tpb, fpb, fnb = b
    if len(pa) != len(pb):
        print("\n%s minus %s  -- skipped, different test sets (%d vs %d images)"
              % (name_a, name_b, len(pa), len(pb)))
        return None
    n = len(pa)
    diffs = {"BLEU-4": [], "ROUGE-L": [], "CE_micro_F1": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs["BLEU-4"].append(bleu4_from_parts(pa, idx) - bleu4_from_parts(pb, idx))
        diffs["ROUGE-L"].append(float(ra[idx].mean() - rb[idx].mean()))
        diffs["CE_micro_F1"].append(micro_f1(tpa, fpa, fna, idx) - micro_f1(tpb, fpb, fnb, idx))

    print("\n%s minus %s" % (name_a, name_b))
    res = {}
    for k, v in diffs.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        print("  %-12s %+0.4f   95%% CI [%+0.4f, %+0.4f]   distinguishable: %s"
              % (k, v.mean(), lo, hi, sig))
        res[k] = {"diff": round(float(v.mean()), 4), "lo": round(float(lo), 4),
                  "hi": round(float(hi), 4), "distinguishable": sig == "yes"}
    return res


def main():
    rng = np.random.default_rng(SEED)
    print("findings measured: %d  (excluded: %s)" % (len(LABELS) - len(UNDETECTABLE),
                                                     ", ".join(UNDETECTABLE)))
    summaries, raws = {}, {}
    for tag in RUNS:
        got = summarise(tag, rng)
        if got is None:
            continue
        summaries[tag], raws[tag] = got

    rows = []
    for tag, s in summaries.items():
        rows.append({"run": tag, "n": s["n"],
                     "BLEU-4": "%.4f [%.4f, %.4f]" % (s["BLEU-4"]["value"], s["BLEU-4"]["lo"], s["BLEU-4"]["hi"]),
                     "ROUGE-L": "%.4f [%.4f, %.4f]" % (s["ROUGE-L"]["value"], s["ROUGE-L"]["lo"], s["ROUGE-L"]["hi"]),
                     "CE micro F1": "%.4f [%.4f, %.4f]" % (s["CE_micro_F1"]["value"], s["CE_micro_F1"]["lo"], s["CE_micro_F1"]["hi"])})
    print("\npoint estimate with 95%% bootstrap CI (%d resamples)" % N_BOOT)
    print(pd.DataFrame(rows).to_string(index=False))

    comparisons = {}
    for a, b in [("oracle224", "none224"), ("oracle224", "uncal224"), ("oracle224", "main224"),
                 ("uncal224", "none224"), ("main224", "none224"), ("main224", "uncal224")]:
        if a in raws and b in raws:
            got = paired(raws[a], raws[b], rng, a, b)
            if got is not None:
                comparisons["%s_vs_%s" % (a, b)] = got

    with open("comparison.json", "w") as f:
        json.dump({"summaries": summaries, "comparisons": comparisons}, f, indent=1)
    print("\nsaved comparison.json")


if __name__ == "__main__":
    main()