"""
Counts how often the generated reports make comparisons with a prior study,
which the model cannot support from a single image.

The word list is kept here rather than imported, so running this never touches
the data files.

    python check_temporal.py
"""

import json
import os
import re
from collections import Counter

TAGS = ["uncal224", "none224", "oracle224", "main224",
        "phrase224", "oracle_phrase", "none_phrase", "cal_phrase",
        "phrase224_s2", "uncal224_s2", "unfilt224"]

TEMPORAL = [
    "compared to", "comparison", "in comparison", "previous", "prior", "earlier",
    "unchanged", "changed", "no change", "little change", "interval", "since",
    "again", "redemonstrat", "re-demonstrat", "persistent", "stable", "constant",
    "improved", "improving", "worsened", "worsening", "progressed", "progression",
    "resolved", "resolving", "decreased", "increased", "new ", "newly", "removed",
    "has been", "have been", "continues", "continue to", "similar to", "as before",
    "no longer", "status post", "postoperative",
]

PAT = re.compile("|".join(re.escape(t) for t in TEMPORAL))


def main():
    print("%-12s %8s %14s %12s" % ("run", "reports", "with a claim", "per report"))
    detail = {}
    for tag in TAGS:
        path = os.path.join("checkpoints_gen_" + tag, "test_generated.json")
        if not os.path.isfile(path):
            continue
        data = json.load(open(path))
        hits, total, words = 0, 0, Counter()
        for d in data:
            found = PAT.findall(d["hyp"].lower())
            if found:
                hits += 1
            total += len(found)
            words.update(found)
        print("%-12s %8d %13.1f%% %12.2f"
              % (tag, len(data), 100 * hits / len(data), total / len(data)))
        detail[tag] = words

    for tag in ["phrase224", "unfilt224"]:
        if tag in detail and sum(detail[tag].values()):
            print("\n%s, most common:" % tag)
            print("   " + ", ".join("%s (%d)" % (w.strip(), c)
                                    for w, c in detail[tag].most_common(8)))

    for tag in TAGS:
        path = os.path.join("checkpoints_gen_" + tag, "test_generated.json")
        if os.path.isfile(path):
            data = json.load(open(path))
            hits = sum(1 for d in data if PAT.search(d["ref"].lower()))
            print("\nreference reports used by %s: %.1f%% contain one"
                  % (tag, 100 * hits / len(data)))


if __name__ == "__main__":
    main()