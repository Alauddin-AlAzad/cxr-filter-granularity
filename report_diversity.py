"""Measures how many distinct reports each trained run produces.

Reads every checkpoints_gen_*/test_generated.json in the current folder and
reports, per run, how much of the output is the same text repeated. A low
distinct rate means the decoder is reproducing templates rather than
describing each image.

    python report_diversity.py
    python report_diversity.py --dir E:/mimic_data --top 5
    python report_diversity.py --csv diversity.csv
"""

import argparse
import csv
import glob
import json
import os
import random
from collections import Counter

RUN_LABEL = {
    "none224": ("none", "sentence"),
    "uncal224": ("predicted", "sentence"),
    "main224": ("calibrated", "sentence"),
    "oracle224": ("oracle", "sentence"),
    "phrase224": ("predicted", "phrase"),
    "none_phrase": ("none", "phrase"),
    "oracle_phrase": ("oracle", "phrase"),
    "unfilt224": ("predicted", "unfiltered"),
    "phrase224_s2": ("predicted", "phrase (seed 2)"),
    "uncal224_s2": ("predicted", "sentence (seed 2)"),
}


def load_hyps(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ("generated", "hyps", "results", "data"):
            if k in data:
                data = data[k]
                break
    out = []
    for d in data:
        if isinstance(d, str):
            out.append(d)
        elif isinstance(d, dict):
            for k in ("hyp", "generated", "pred", "report", "text"):
                if k in d:
                    out.append(str(d[k]))
                    break
    return out


def norm(s):
    return " ".join(str(s).lower().split())


def stats(hyps):
    texts = [norm(h) for h in hyps]
    n = len(texts)
    c = Counter(texts)
    top = c.most_common()
    shared = sum(v for v in c.values() if v > 1)
    lens = [len(t.split()) for t in texts] or [0]
    return {
        "n": n,
        "distinct": len(c),
        "distinct_pct": 100.0 * len(c) / n if n else 0.0,
        "top1_n": top[0][1] if top else 0,
        "top1_pct": 100.0 * top[0][1] / n if n else 0.0,
        "top5_pct": 100.0 * sum(v for _, v in top[:5]) / n if n else 0.0,
        "shared_pct": 100.0 * shared / n if n else 0.0,
        "singletons": sum(1 for v in c.values() if v == 1),
        "mean_words": sum(lens) / len(lens),
        "top": top,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--top", type=int, default=3, help="example templates to print")
    ap.add_argument("--csv", default="")
    ap.add_argument("--truncate", type=int, default=0,
                    help="compare only the first N words of each report")
    ap.add_argument("--match-n", action="store_true",
                    help="subsample every run to the smallest n, same seed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pattern = os.path.join(args.dir, "checkpoints_gen_*", "test_generated.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no files matched: {pattern}")
        print("run this from the folder that holds the checkpoints_gen_* folders,")
        print("or pass --dir")
        return

    loaded = []
    for p in paths:
        tag = os.path.basename(os.path.dirname(p)).replace("checkpoints_gen_", "")
        hyps = load_hyps(p)
        if not hyps:
            print(f"{tag}: could not read any reports from {p}")
            continue
        loaded.append((tag, hyps))
    if not loaded:
        return

    if args.match_n:
        m = min(len(h) for _, h in loaded)
        trimmed = []
        for tag, hyps in loaded:
            if len(hyps) > m:
                idx = sorted(random.Random(args.seed).sample(range(len(hyps)), m))
                hyps = [hyps[i] for i in idx]
            trimmed.append((tag, hyps))
        loaded = trimmed
        print(f"\n[match-n] every run subsampled to n = {m} (seed {args.seed})")
    if args.truncate:
        loaded = [(tag, [" ".join(str(h).split()[: args.truncate]) for h in hyps])
                  for tag, hyps in loaded]
        print(f"[truncate] comparing only the first {args.truncate} words of each report")

    rows = []
    for tag, hyps in loaded:
        s = stats(hyps)
        s["tag"] = tag
        prompt, text = RUN_LABEL.get(tag, ("?", "?"))
        s["prompt"], s["train_text"] = prompt, text
        rows.append(s)

    if not rows:
        return

    rows.sort(key=lambda r: r["distinct_pct"])

    print(f"\n{'run':16s}{'prompt':11s}{'text':16s}{'n':>6s}{'distinct':>10s}"
          f"{'%':>8s}{'top1 %':>9s}{'top5 %':>9s}{'repeated %':>12s}{'words':>8s}")
    print("-" * 105)
    for r in rows:
        print(f"{r['tag']:16s}{r['prompt']:11s}{r['train_text']:16s}{r['n']:>6d}"
              f"{r['distinct']:>10d}{r['distinct_pct']:>8.1f}{r['top1_pct']:>9.1f}"
              f"{r['top5_pct']:>9.1f}{r['shared_pct']:>12.1f}{r['mean_words']:>8.1f}")

    print("\ncolumn meanings")
    print("  distinct %   share of outputs that are a one-off text")
    print("  top1 %       share of all images that received the single commonest text")
    print("  top5 %       share covered by the five commonest texts")
    print("  repeated %   share of images whose text also went to another image")

    if args.top:
        print("\n" + "=" * 70)
        for r in rows:
            print(f"\n{r['tag']}  ({r['prompt']} prompt, {r['train_text']} text)")
            for t, c in r["top"][:args.top]:
                print(f"  x{c:<5d} {t[:88]}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "prompt", "train_text", "n", "distinct", "distinct_pct",
                        "top1_n", "top1_pct", "top5_pct", "repeated_pct", "singletons",
                        "mean_words"])
            for r in rows:
                w.writerow([r["tag"], r["prompt"], r["train_text"], r["n"], r["distinct"],
                            round(r["distinct_pct"], 2), r["top1_n"], round(r["top1_pct"], 2),
                            round(r["top5_pct"], 2), round(r["shared_pct"], 2),
                            r["singletons"], round(r["mean_words"], 1)])
        print(f"\nwritten: {os.path.abspath(args.csv)}")


main()