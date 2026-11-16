"""
Runs the paired comparisons that compare_runs.py does not cover.

compare_runs.py only compares the four sentence-level arms, so the phrase-level
contrasts have a point difference but no interval. This fills them in, reusing
the same bootstrap so the numbers are directly comparable.

    python extra_pairs.py

Writes extra_pairs.json.
"""

import json

import numpy as np

from compare_runs import summarise, paired, N_BOOT

PAIRS = [
    ("oracle_phrase", "phrase224"),
    ("oracle_phrase", "none_phrase"),
    ("phrase224", "none_phrase"),
    ("phrase224_s2", "uncal224_s2"),
    ("phrase224", "phrase224_s2"),
    ("uncal224", "uncal224_s2"),
]
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    raws = {}
    for tag in sorted({t for p in PAIRS for t in p}):
        got = summarise(tag, rng)
        if got is None:
            print("missing run: %s" % tag)
            continue
        _, raws[tag] = got

    out = {}
    for a, b in PAIRS:
        if a in raws and b in raws:
            res = paired(raws[a], raws[b], rng, a, b)
            if res is not None:
                out["%s_vs_%s" % (a, b)] = res

    with open("extra_pairs.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved extra_pairs.json  (%d bootstrap resamples)" % N_BOOT)


if __name__ == "__main__":
    main()