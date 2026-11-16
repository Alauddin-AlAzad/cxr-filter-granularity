"""Rebuild incxr_radiologist_key.csv from the manifest.

The original key was written with
    truth = "normal" if name.endswith("normal") else "abnormal"
and "abnormal" also ends with "normal", so every row was labelled normal.
The image selection was correct; only the label column was wrong, so the
labels can be recovered from the manifest without rerunning anything.

    python fix_key.py incxr_radiologist_key.csv
    python fix_key.py incxr_radiologist_key.csv path/to/manifest.csv
"""

import csv
import os
import sys
from collections import Counter

DEFAULTS = [
    os.path.join("external", "in_cxr", "manifest.csv"),
    "manifest.csv",
]


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def find_manifest():
    for p in DEFAULTS:
        if os.path.exists(p):
            return p
    for root, _, files in os.walk("."):
        if "manifest.csv" in files:
            return os.path.join(root, "manifest.csv")
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python fix_key.py <key.csv> [manifest.csv]")
    key_path = sys.argv[1]
    man_path = sys.argv[2] if len(sys.argv) > 2 else find_manifest()
    if not os.path.exists(key_path):
        sys.exit(f"key file not found: {key_path}")
    if not man_path or not os.path.exists(man_path):
        sys.exit("manifest.csv not found. Pass its path:\n"
                 "  python fix_key.py incxr_radiologist_key.csv <path to manifest.csv>")

    key_rows = load(key_path)
    man_rows = load(man_path)
    print(f"key rows: {len(key_rows)}   manifest rows: {len(man_rows)}")
    print(f"manifest: {os.path.abspath(man_path)}")
    print(f"manifest columns: {list(man_rows[0].keys())}\n")

    if "abnormal" not in man_rows[0]:
        sys.exit("the manifest has no 'abnormal' column; cannot recover the labels")
    if "image_id" not in key_rows[0]:
        sys.exit(f"the key has no 'image_id' column; it has {list(key_rows[0].keys())}")

    truth = {}
    for r in man_rows:
        v = str(r["abnormal"]).strip()
        truth[str(r["image_id"]).strip()] = "abnormal" if v in {"1", "1.0", "True", "true"} else "normal"

    print("old key labels:", dict(Counter(r.get("truth", "") for r in key_rows)))

    out, missing = [], []
    for r in key_rows:
        img = str(r["image_id"]).strip()
        if img not in truth:
            missing.append(r.get("case", img))
            continue
        out.append({"case": r.get("case", ""), "image_id": img, "truth": truth[img]})

    if missing:
        print(f"\nno manifest entry for {len(missing)} cases: {missing}")
    print("new key labels:", dict(Counter(r["truth"] for r in out)))

    dest = "incxr_radiologist_key_fixed.csv"
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "image_id", "truth"])
        w.writeheader()
        w.writerows(out)
    print(f"\nwritten: {os.path.abspath(dest)}")
    print("next:  python rating_check.py cxr_ratings_3.csv incxr_radiologist_key_fixed.csv")


main()