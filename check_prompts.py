"""
Checks why some findings never appear in the generated reports.

Two possible causes:
  a) the finding is dropped from the prompt by the three-finding cap
  b) the finding reaches the prompt but the model never writes the words

Counts both, so the next run is only spent if (a) is the real one.

    python check_prompts.py
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from cxr_data import LABEL_COLS
from gen_data import load_splits, MAX_FINDINGS
from clinical_eval import extract_findings, LABELS

DATA_DIR = "."
DISEASES = LABEL_COLS[1:]


def prompt_stats(df, name):
    n = len(df)
    in_prompt = Counter()
    positive = Counter()
    dropped = Counter()
    for _, r in df.iterrows():
        pos = [c for c in DISEASES if r.get(c, 0) == 1]
        for c in pos:
            positive[c] += 1
        for c in r["findings"]:
            in_prompt[c] += 1
        for c in pos:
            if c not in r["findings"]:
                dropped[c] += 1

    rows = []
    for c in DISEASES:
        p = positive[c]
        rows.append({"finding": c,
                     "positive": p,
                     "in_prompt": in_prompt[c],
                     "dropped_by_cap": dropped[c],
                     "kept_%": round(100 * in_prompt[c] / p, 1) if p else 0.0})
    print("\n%s  (%d images, cap = %d findings)" % (name, n, MAX_FINDINGS))
    print(pd.DataFrame(rows).to_string(index=False))

    over = np.mean([sum(1 for c in DISEASES if r.get(c, 0) == 1) > MAX_FINDINGS
                    for _, r in df.iterrows()])
    print("  images with more than %d true findings: %.1f%%" % (MAX_FINDINGS, 100 * over))


def generated_stats(path, name):
    if not os.path.isfile(path):
        print("\nnot found:", path)
        return
    data = json.load(open(path))
    named = Counter()
    for d in data:
        for k, v in extract_findings(d["hyp"]).items():
            if v:
                named[k] += 1
    print("\n%s  (%d reports)" % (name, len(data)))
    for c in LABELS:
        print("  %-28s written in %5d reports  (%.1f%%)"
              % (c, named[c], 100 * named[c] / max(len(data), 1)))


def main():
    for prompts in ["calibrated", "oracle"]:
        tr, va, te = load_splits(DATA_DIR, prompts, "filtered")
        prompt_stats(te, "test prompts, %s" % prompts)

    generated_stats("checkpoints_gen_main224/test_generated.json", "generated with predicted prompts")
    generated_stats("checkpoints_gen_oracle224/test_generated.json", "generated with oracle prompts")


if __name__ == "__main__":
    main()