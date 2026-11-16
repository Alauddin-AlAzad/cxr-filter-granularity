"""
Sweeps beam width and length penalty on the VALIDATION split.

The decoder caps generation at 100 tokens and uses beam 3 with no length
preference, which produces reports shorter than the references. Raising the
length penalty makes the search prefer longer output, which should raise recall
at some cost in precision. Nothing is retrained; only decoding changes.

Validation only, on purpose. Choosing these values on the test set would be
tuning on the test set. Pick one setting here, then apply it to every run.

    python tune_decoding.py
    python tune_decoding.py --tag phrase224 --n 300

Writes tune_decoding_<tag>.json.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from gen_data import ReportDataset, Vocab, collate, make_transforms, load_splits
from clinical_eval import extract_findings, LABELS
from train_generator import PromptedGenerator

UNDETECTABLE = ["Enlarged Cardiomediastinum"]
RUNS = {
    "phrase224": ("uncalibrated", "phrase"),
    "uncal224": ("uncalibrated", "filtered"),
    "none_phrase": ("none", "phrase"),
}
BEAMS = [3, 5]
PENALTIES = [1.0, 1.2, 1.5, 2.0]
IMG_SIZE = 224


def micro_f1(pred, true):
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@torch.no_grad()
def run(model, vocab, df, device, beam, penalty, max_len):
    loader = DataLoader(ReportDataset(df, vocab, make_transforms(IMG_SIZE, False)),
                        batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)
    hyps = []
    for b in loader:
        out = model.generate(b["img"].to(device), b["prompt"].to(device),
                             beam=beam, length_penalty=penalty, max_len=max_len)
        for row in out:
            hyps.append(" ".join(vocab.decode(row.cpu())))
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="phrase224")
    ap.add_argument("--n", type=int, default=300, help="validation images to use")
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.tag not in RUNS:
        print("tag must be one of %s" % ", ".join(RUNS))
        return
    prompts, reports = RUNS[args.tag]
    ckpt_dir = "checkpoints_gen_" + args.tag
    if not os.path.isfile(os.path.join(ckpt_dir, "best.pt")):
        print("no checkpoint at %s" % ckpt_dir)
        return

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    _, va, _ = load_splits(".", prompts, reports)
    va = va.iloc[: args.n].reset_index(drop=True)
    labels = [l for l in LABELS if l not in UNDETECTABLE]
    true = va[labels].fillna(0).replace(-1, 0).values.astype(int).astype(bool)

    vocab = Vocab.load(os.path.join(ckpt_dir, "vocab.json"))
    model = PromptedGenerator(len(vocab), None, n_tokens=(IMG_SIZE // 32) ** 2).to(device)
    state = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    print("run: %s   validation images: %d   max_len: %d\n" % (args.tag, len(va), args.max_len))
    print("%-5s %-9s %8s %8s %8s %8s" % ("beam", "penalty", "words", "prec", "recall", "F1"))

    rows = []
    for beam in BEAMS:
        for pen in PENALTIES:
            hyps = run(model, vocab, va, device, beam, pen, args.max_len)
            pred = np.zeros_like(true)
            for i, h in enumerate(hyps):
                f = extract_findings(h)
                for j, lab in enumerate(labels):
                    pred[i, j] = bool(f.get(lab, 0))
            p, r, f1 = micro_f1(pred, true)
            words = float(np.mean([len(h.split()) for h in hyps]))
            print("%-5d %-9.1f %8.1f %8.3f %8.3f %8.3f" % (beam, pen, words, p, r, f1))
            rows.append({"beam": beam, "length_penalty": pen, "mean_words": round(words, 1),
                         "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})

    base = next(x for x in rows if x["beam"] == 3 and x["length_penalty"] == 1.0)
    best = max(rows, key=lambda x: x["f1"])
    print("\ncurrent setting (beam 3, penalty 1.0): F1 %.4f" % base["f1"])
    print("best on validation: beam %d, penalty %.1f, F1 %.4f  (%+.4f)"
          % (best["beam"], best["length_penalty"], best["f1"], best["f1"] - base["f1"]))
    if best["f1"] - base["f1"] < 0.01:
        print("\nthe gain is under 0.01 - not worth regenerating every run")

    with open("tune_decoding_%s.json" % args.tag, "w") as f:
        json.dump({"rows": rows, "n": len(va), "max_len": args.max_len}, f, indent=1)
    print("\nsaved tune_decoding_%s.json" % args.tag)


if __name__ == "__main__":
    main()