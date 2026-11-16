"""
Compares the unfiltered run with the filtered one on exactly the same images.

unfilt224 was evaluated on 1302 images and uncal224 on 1227, so the headline
difference could come from the 75 extra images rather than from the training
data. This restricts both to the images they share and repeats the comparison,
including a paired bootstrap, which is only valid once the sets match.

    python check_unfilt.py

Writes unfilt_check.json.
"""

import json
import os

import numpy as np

from clinical_eval import LABELS
from compare_runs import (N_BOOT, SEED, UNDETECTABLE, bleu4_from_parts, bleu4_parts,
                          ce_parts, micro_f1, rouge_l_pair)
from gen_data import load_splits

LABELS_USED = [l for l in LABELS if l not in UNDETECTABLE]


def load(tag, prompts, reports):
    path = os.path.join("checkpoints_gen_" + tag, "test_generated.json")
    data = json.load(open(path))
    _, _, te = load_splits(".", prompts, reports)
    n = min(len(data), len(te))
    te = te.iloc[:n].reset_index(drop=True)
    by_id = {}
    for i in range(n):
        by_id[te.loc[i, "dicom_id"]] = (data[i]["hyp"], data[i]["ref"],
                                        te.loc[i, LABELS_USED].fillna(0).replace(-1, 0).values.astype(int))
    return by_id


def parts_for(ids, by_id):
    hyps = [by_id[i][0] for i in ids]
    refs = [by_id[i][1] for i in ids]
    true = np.array([by_id[i][2] for i in ids])
    h = [x.lower().split() for x in hyps]
    r = [x.lower().split() for x in refs]
    return {"hyps": hyps,
            "bleu": [bleu4_parts(a, b) for a, b in zip(h, r)],
            "rouge": np.array([rouge_l_pair(a, b) for a, b in zip(h, r)]),
            "ce": ce_parts(hyps, true, LABELS_USED),
            "len": float(np.mean([len(x) for x in h]))}


def summarise(name, p, rng):
    n = len(p["hyps"])
    full = np.arange(n)
    tp, fp, fn = p["ce"]
    point = {"BLEU-4": bleu4_from_parts(p["bleu"], full),
             "ROUGE-L": float(p["rouge"].mean()),
             "CE_micro_F1": micro_f1(tp, fp, fn, full)}
    out = {"n": n, "mean_words": round(p["len"], 1)}
    for k, v in point.items():
        boot = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            if k == "BLEU-4":
                boot.append(bleu4_from_parts(p["bleu"], idx))
            elif k == "ROUGE-L":
                boot.append(float(p["rouge"][idx].mean()))
            else:
                boot.append(micro_f1(tp, fp, fn, idx))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[k] = {"value": round(v, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
    print("  %-12s n=%4d  mean words %5.1f   BLEU-4 %.4f   ROUGE-L %.4f   CE F1 %.4f"
          % (name, n, out["mean_words"], point["BLEU-4"], point["ROUGE-L"], point["CE_micro_F1"]))
    return out


def paired(a, b, rng, name_a, name_b):
    n = len(a["hyps"])
    tpa, fpa, fna = a["ce"]
    tpb, fpb, fnb = b["ce"]
    print("\n%s minus %s  (same %d images)" % (name_a, name_b, n))
    res = {}
    diffs = {"BLEU-4": [], "ROUGE-L": [], "CE_micro_F1": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs["BLEU-4"].append(bleu4_from_parts(a["bleu"], idx) - bleu4_from_parts(b["bleu"], idx))
        diffs["ROUGE-L"].append(float(a["rouge"][idx].mean() - b["rouge"][idx].mean()))
        diffs["CE_micro_F1"].append(micro_f1(tpa, fpa, fna, idx) - micro_f1(tpb, fpb, fnb, idx))
    for k, v in diffs.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        print("  %-12s %+0.4f   95%% CI [%+0.4f, %+0.4f]   distinguishable: %s"
              % (k, v.mean(), lo, hi, sig))
        res[k] = {"diff": round(float(v.mean()), 4), "lo": round(float(lo), 4),
                  "hi": round(float(hi), 4), "distinguishable": sig == "yes"}
    return res


RUNS = [("uncal224", "uncalibrated", "filtered"),
        ("none224", "none", "filtered"),
        ("oracle224", "oracle", "filtered"),
        ("phrase224", "uncalibrated", "phrase"),
        ("oracle_phrase", "oracle", "phrase"),
        ("none_phrase", "none", "phrase"),
        ("cal_phrase", "calibrated", "phrase"),
        ("unfilt224", "uncalibrated", "unfiltered"),
        ("phrase224_s2", "uncalibrated", "phrase"),
        ("uncal224_s2", "uncalibrated", "filtered")]

PAIRS = [("phrase224", "uncal224"), ("unfilt224", "uncal224"),
         ("phrase224", "unfilt224"),
         ("oracle_phrase", "phrase224"), ("oracle_phrase", "oracle224"),
         ("oracle224", "uncal224"),
         ("phrase224", "none_phrase"), ("none224", "uncal224"),
         ("phrase224_s2", "uncal224_s2"),
         ("phrase224", "phrase224_s2"), ("uncal224", "uncal224_s2")]


def main():
    loaded = {}
    for tag, prompts, reports in RUNS:
        if os.path.isfile(os.path.join("checkpoints_gen_" + tag, "test_generated.json")):
            loaded[tag] = load(tag, prompts, reports)
        else:
            print("not found, skipping:", tag)

    common = None
    for by_id in loaded.values():
        common = set(by_id) if common is None else common & set(by_id)
    common = sorted(common or [])
    print("images per run: %s"
          % ", ".join("%s %d" % (t, len(b)) for t, b in loaded.items()))
    print("shared by all: %d" % len(common))
    if not common:
        return

    rng = np.random.default_rng(SEED)
    parts = {t: parts_for(common, b) for t, b in loaded.items()}

    print("\non the shared images only:")
    out = {"shared": len(common)}
    for tag in loaded:
        out[tag] = summarise(tag, parts[tag], rng)

    comps = {}
    for a, b in PAIRS:
        if a in parts and b in parts:
            comps["%s_vs_%s" % (a, b)] = paired(parts[a], parts[b], rng, a, b)

    out["comparisons"] = comps
    with open("unfilt_check.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved unfilt_check.json")


if __name__ == "__main__":
    main()