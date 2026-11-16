"""
Breaks BLEU-4 into its parts, to show where the score is actually lost.

BLEU-4 = BP * exp(mean(log p1..p4)), so a low score has only two possible
causes: short output (BP below 1) or low n-gram precision. This prints both,
plus the length statistics behind the penalty.

    python diag_bleu.py
    python diag_bleu.py --tag uncal224

Reads checkpoints_gen_<tag>/test_generated.json. No GPU needed.
"""

import argparse
import json
import math
import os
from collections import Counter

import numpy as np


def ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_parts(hyps, refs):
    num = [0] * 5
    den = [0] * 5
    hl = rl = 0
    for h, r in zip(hyps, refs):
        hl += len(h)
        rl += len(r)
        for n in range(1, 5):
            hn, rn = ngrams(h, n), ngrams(r, n)
            num[n] += sum(min(c, rn[g]) for g, c in hn.items())
            den[n] += sum(hn.values())
    return num, den, hl, rl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="phrase224")
    args = ap.parse_args()

    path = os.path.join("checkpoints_gen_" + args.tag, "test_generated.json")
    if not os.path.isfile(path):
        print("not found:", path)
        return
    data = json.load(open(path))
    hyps = [d["hyp"].lower().split() for d in data]
    refs = [d["ref"].lower().split() for d in data]

    num, den, hl, rl = corpus_parts(hyps, refs)
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    p = [0.0] * 5
    for n in range(1, 5):
        p[n] = num[n] / den[n] if den[n] else 0.0
    logs = [math.log(p[n]) for n in range(1, 5) if p[n] > 0]
    geo = math.exp(sum(logs) / 4) if len(logs) == 4 else 0.0
    bleu = bp * geo

    hw = np.array([len(h) for h in hyps])
    rw = np.array([len(r) for r in refs])

    print("run: %s   pairs: %d\n" % (args.tag, len(hyps)))

    print("length")
    print("  generated   mean %.1f  median %.0f  min %d  max %d" % (hw.mean(), np.median(hw), hw.min(), hw.max()))
    print("  reference   mean %.1f  median %.0f  min %d  max %d" % (rw.mean(), np.median(rw), rw.min(), rw.max()))
    print("  total words: generated %d, reference %d  (ratio %.3f)" % (hl, rl, hl / max(rl, 1)))
    print("  brevity penalty: %.4f%s" % (bp, "" if bp >= 0.999 else "   <- this multiplies the score"))

    print("\nn-gram precision")
    for n in range(1, 5):
        print("  p%d = %6d / %6d = %.4f" % (n, num[n], den[n], p[n]))
    print("  geometric mean of p1..p4: %.4f" % geo)

    print("\nBLEU-4 = %.4f x %.4f = %.4f" % (bp, geo, bleu))

    print("\nwhat it would be if the other part were perfect")
    print("  with BP = 1.0 (length matched):      %.4f" % geo)
    if geo > 0:
        print("  share of the loss from length:       %.0f%%" % (100 * (1 - bp)))

    # how much of the output is the same sentence over and over
    first = Counter(" ".join(h[:8]) for h in hyps)
    top, cnt = first.most_common(1)[0]
    uniq = len(set(" ".join(h) for h in hyps))
    print("\nrepetition")
    print("  distinct reports: %d of %d  (%.1f%%)" % (uniq, len(hyps), 100 * uniq / len(hyps)))
    print("  most common opening (8 words): %d reports  (%.1f%%)" % (cnt, 100 * cnt / len(hyps)))
    print("    \"%s\"" % top[:90])


if __name__ == "__main__":
    main()