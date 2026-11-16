"""
CheXbert F1 on the saved generated reports, with bootstrap confidence intervals.

The rule-based extractor in clinical_eval.py only matches words, so a report
that repeats the prompt scores well whether or not the finding is asserted.
CheXbert is a BERT labeller trained on radiology reports and reads negation and
hedging, so it is the fairer measure and the one the papers quote.

    python chexbert_eval.py

Per-image labels are cached in chexbert_labels.json, so the first run labels
every report (slow, CPU) and later runs reuse them and only resample.

Writes chexbert.json and chexbert_labels.json.
"""

import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

REPO = "StanfordAIMI/RRG_scorers"
MIN_SIZE = 100 * 1024 * 1024
LABEL_CACHE = "chexbert_labels.json"
N_BOOT = 1000
SEED = 0

TARGETS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
           "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
           "Pleural Other", "Fracture", "Support Devices", "No Finding"]
TARGETS_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
IDX_5 = [TARGETS.index(c) for c in TARGETS_5]

# 2-epoch diagnostic run, not comparable with the 8-epoch runs
EXCLUDE_FROM_PAIRS = {"probe_ls0"}

PAIRS = [("oracle224", "none224"), ("oracle224", "uncal224"), ("oracle224", "main224"),
         ("uncal224", "none224"), ("main224", "none224"), ("main224", "uncal224")]


def ensure_checkpoint():
    from appdirs import user_cache_dir
    from huggingface_hub import hf_hub_download

    cache_dir = user_cache_dir("chexbert")
    target = os.path.join(cache_dir, "chexbert.pth")
    if os.path.isfile(target) and os.path.getsize(target) > MIN_SIZE:
        return target

    found = None
    for root, _, files in os.walk(cache_dir):
        if "chexbert.pth" in files:
            path = os.path.join(root, "chexbert.pth")
            if os.path.getsize(path) > MIN_SIZE:
                found = path
                break

    if found is None:
        print("checkpoint not in cache, downloading")
        os.makedirs(cache_dir, exist_ok=True)
        found = hf_hub_download(repo_id=REPO, filename="chexbert.pth", cache_dir=cache_dir)

    print("checkpoint:", found)
    try:
        os.link(found, target)
    except OSError:
        shutil.copy(found, target)
    return target


class TokenizerShim:
    """f1chexbert calls encode_plus on a pre-tokenised list; transformers 5
    removed that method, so this supplies the two calls it actually needs."""

    def __init__(self, tok):
        self.tok = tok
        self.cls_token_id = tok.cls_token_id
        self.sep_token_id = tok.sep_token_id

    def tokenize(self, text):
        return self.tok.tokenize(text)

    def encode_plus(self, tokens):
        ids = self.tok.convert_tokens_to_ids(tokens)
        return {"input_ids": [self.cls_token_id] + ids + [self.sep_token_id]}


def build_scorer():
    ensure_checkpoint()
    from f1chexbert import F1CheXbert
    scorer = F1CheXbert(device="cpu")
    if not hasattr(scorer.tokenizer, "encode_plus"):
        scorer.tokenizer = TokenizerShim(scorer.tokenizer)
    return scorer


def label_all(scorer, texts, what):
    out = []
    for i, t in enumerate(texts):
        out.append([int(v) for v in scorer.get_label(t.strip())])
        if (i + 1) % 300 == 0:
            print("    %s %d/%d" % (what, i + 1, len(texts)), flush=True)
    return out


# ---------------------------------------------------------------- scoring


def counts(hyp, ref, cols):
    """Per-image tp/fp/fn over the chosen label columns, so a bootstrap can
    re-pool them without relabelling."""
    h = np.asarray(hyp)[:, cols]
    r = np.asarray(ref)[:, cols]
    tp = ((h == 1) & (r == 1)).sum(axis=1)
    fp = ((h == 1) & (r == 0)).sum(axis=1)
    fn = ((h == 0) & (r == 1)).sum(axis=1)
    return tp, fp, fn


def prf(tp, fp, fn, idx):
    a, b, c = tp[idx].sum(), fp[idx].sum(), fn[idx].sum()
    p = a / (a + b) if a + b else 0.0
    r = a / (a + c) if a + c else 0.0
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f


def ci(tp, fp, fn, rng):
    n = len(tp)
    point = prf(tp, fp, fn, np.arange(n))[2]
    boot = [prf(tp, fp, fn, rng.integers(0, n, n))[2] for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def per_class_f1(hyp, ref):
    h, r = np.asarray(hyp), np.asarray(ref)
    out = {}
    for j, name in enumerate(TARGETS):
        tp = int(((h[:, j] == 1) & (r[:, j] == 1)).sum())
        fp = int(((h[:, j] == 1) & (r[:, j] == 0)).sum())
        fn = int(((h[:, j] == 0) & (r[:, j] == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {"f1": round(0.0 if p + rc == 0 else 2 * p * rc / (p + rc), 4),
                     "support": tp + fn}
    return out


def paired(a, b, rng, name_a, name_b):
    """Resamples the same image indices for both runs. Only valid when the two
    runs were evaluated on the same images in the same order."""
    if len(a[0][0]) != len(b[0][0]):
        print("\n%s minus %s  -- skipped, different test sets" % (name_a, name_b))
        return None
    n = len(a[0][0])
    print("\n%s minus %s" % (name_a, name_b))
    res = {}
    for key, (ca, cb) in [("14-label", (a[0], b[0])), ("5-label", (a[1], b[1]))]:
        d = [prf(*ca, idx)[2] - prf(*cb, idx)[2]
             for idx in (rng.integers(0, n, n) for _ in range(N_BOOT))]
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        print("  %-10s %+0.4f   95%% CI [%+0.4f, %+0.4f]   distinguishable: %s"
              % (key, d.mean(), lo, hi, sig))
        res[key] = {"diff": round(float(d.mean()), 4), "lo": round(float(lo), 4),
                    "hi": round(float(hi), 4), "distinguishable": sig == "yes"}
    return res


# ---------------------------------------------------------------- main


def main():
    paths = sorted(glob.glob(os.path.join("checkpoints_gen_*", "test_generated.json")))
    if not paths:
        print("no test_generated.json found")
        return

    cache = {}
    if os.path.isfile(LABEL_CACHE):
        cache = json.load(open(LABEL_CACHE))
        print("label cache: %d runs" % len(cache))

    runs = {}
    for path in paths:
        tag = os.path.basename(os.path.dirname(path)).replace("checkpoints_gen_", "")
        data = json.load(open(path))
        runs[tag] = ([d["hyp"] for d in data], [d["ref"] for d in data])

    todo = [t for t, (h, _) in runs.items()
            if t not in cache or len(cache[t]["hyp"]) != len(h)]

    if todo:
        scorer = build_scorer()
        ref_seen = {}
        for tag in todo:
            hyps, refs = runs[tag]
            print("labelling %s (%d reports)" % (tag, len(hyps)), flush=True)
            key = str(hash(tuple(refs)))
            if key not in ref_seen:
                ref_seen[key] = label_all(scorer, refs, "reference")
            else:
                print("    reference labels reused", flush=True)
            cache[tag] = {"hyp": label_all(scorer, hyps, "generated"),
                          "ref": ref_seen[key]}
            with open(LABEL_CACHE, "w") as f:
                json.dump(cache, f)
        print("saved %s" % LABEL_CACHE)
    else:
        print("all runs cached, no labelling needed")

    rng = np.random.default_rng(SEED)
    out, raws, rows = {}, {}, []

    for tag in sorted(runs):
        hyp, ref = cache[tag]["hyp"], cache[tag]["ref"]
        c14 = counts(hyp, ref, list(range(len(TARGETS))))
        c5 = counts(hyp, ref, IDX_5)
        raws[tag] = (c14, c5)

        f14, lo14, hi14 = ci(*c14, rng)
        f5, lo5, hi5 = ci(*c5, rng)
        p14, r14, _ = prf(*c14, np.arange(len(hyp)))

        out[tag] = {"n": len(hyp),
                    "all14": {"precision": round(p14, 4), "recall": round(r14, 4),
                              "f1": round(f14, 4), "lo": round(lo14, 4), "hi": round(hi14, 4)},
                    "top5": {"f1": round(f5, 4), "lo": round(lo5, 4), "hi": round(hi5, 4)},
                    "per_class_14": per_class_f1(hyp, ref)}

        rows.append({"run": tag, "n": len(hyp),
                     "14-label F1": "%.4f [%.4f, %.4f]" % (f14, lo14, hi14),
                     "5-label F1": "%.4f [%.4f, %.4f]" % (f5, lo5, hi5),
                     "P": "%.4f" % p14, "R": "%.4f" % r14})

    print("\nCheXbert micro F1 with 95%% bootstrap CI (%d resamples)" % N_BOOT)
    print(pd.DataFrame(rows).to_string(index=False))

    comparisons = {}
    for a, b in PAIRS:
        if a in raws and b in raws and a not in EXCLUDE_FROM_PAIRS and b not in EXCLUDE_FROM_PAIRS:
            got = paired(raws[a], raws[b], rng, a, b)
            if got is not None:
                comparisons["%s_vs_%s" % (a, b)] = got

    with open("chexbert.json", "w") as f:
        json.dump({"summaries": out, "comparisons": comparisons}, f, indent=1)
    print("\nsaved chexbert.json")


if __name__ == "__main__":
    main()