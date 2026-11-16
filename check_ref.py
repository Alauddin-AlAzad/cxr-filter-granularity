import json
from collections import Counter
from clinical_eval import extract_findings, LABELS

data = json.load(open("checkpoints_gen_oracle224/test_generated.json"))
ref = Counter()
for d in data:
    for k, v in extract_findings(d["ref"]).items():
        if v:
            ref[k] += 1

print("found in the REFERENCE reports:")
for c in LABELS:
    print("  %-28s %5d / %d" % (c, ref[c], len(data)))