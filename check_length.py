"""
Checks how much of the unfiltered run's gain is just longer reports.

Truncates its generated reports to the length the filtered run produces and
recomputes clinical efficacy. If the gain survives truncation, length is not
the explanation.

    python check_length.py
"""

import json

import numpy as np

from clinical_eval import LABELS
from compare_runs import N_BOOT, SEED, UNDETECTABLE, ce_parts, micro_f1
from gen_data import load_splits

LABELS_USED = [l for l in LABELS if l not in UNDETECTABLE]


def load(tag, prompts, reports):
    data = json.load(open("checkpoints_gen_%s/test_generated.json" % tag))
    _, _, te = load_splits(".", prompts, reports)
    n = min(len(data), len(te))
    te = te.iloc[:n].reset_index(drop=True)
    return {te.loc[i, "dicom_id"]: (data[i]["hyp"],
                                    te.loc[i, LABELS_USED].fillna(0).replace(-1, 0).values.astype(int))
            for i in range(n)}


def score(hyps, true, rng, name):
    tp, fp, fn = ce_parts(hyps, true, LABELS_USED)
    n = len(hyps)
    point = micro_f1(tp, fp, fn, np.arange(n))
    boot = [micro_f1(tp, fp, fn, rng.integers(0, n, n)) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    words = np.mean([len(h.split()) for h in hyps])
    print("  %-26s mean words %5.1f   CE F1 %.4f  [%.4f, %.4f]"
          % (name, words, point, lo, hi))
    return point


def main():
    unf = load("unfilt224", "uncalibrated", "unfiltered")
    fil = load("uncal224", "uncalibrated", "filtered")
    common = sorted(set(unf) & set(fil))
    true = np.array([fil[i][1] for i in common])

    fil_hyps = [fil[i][0] for i in common]
    unf_hyps = [unf[i][0] for i in common]
    target = int(round(np.mean([len(h.split()) for h in fil_hyps])))

    rng = np.random.default_rng(SEED)
    print("shared images: %d   truncating to %d words" % (len(common), target))
    score(fil_hyps, true, rng, "uncal224 as generated")
    score(unf_hyps, true, rng, "unfilt224 as generated")
    cut = [" ".join(h.split()[:target]) for h in unf_hyps]
    score(cut, true, rng, "unfilt224 truncated")


if __name__ == "__main__":
    main()