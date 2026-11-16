"""
Removes sentences a model cannot infer from a single image (comparisons with prior
studies, change-over-time statements) and optionally keeps FINDINGS only.

Input : image_report_pairs.csv  (renamed to image_report_pairs_raw.csv on first run)
Output: image_report_pairs.csv  (cleaned, so the notebook needs no change)
"""

import os
import re

import pandas as pd

DATA_DIR = "."
FINDINGS_ONLY = False
MIN_WORDS = 4

TEMPORAL = [
    "compared to", "comparison", "in comparison", "previous", "prior", "earlier",
    "unchanged", "changed", "no change", "little change", "interval", "since",
    "again", "redemonstrat", "re-demonstrat", "persistent", "stable", "constant",
    "improved", "improving", "worsened", "worsening", "progressed", "progression",
    "resolved", "resolving", "decreased", "increased", "new ", "newly", "removed",
    "has been", "have been", "continues", "continue to", "similar to", "as before",
    "no longer", "status post", "postoperative",
]

_TEMPORAL_RE = re.compile("|".join(re.escape(t) for t in TEMPORAL))
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def strip_temporal(text):
    kept = [s for s in _SENT_SPLIT.split(str(text)) if s and not _TEMPORAL_RE.search(s.lower())]
    return " ".join(kept).strip()


raw_path = os.path.join(DATA_DIR, "image_report_pairs_raw.csv")
cur_path = os.path.join(DATA_DIR, "image_report_pairs.csv")

if not os.path.isfile(raw_path):
    os.rename(cur_path, raw_path)
    print("Original saved as image_report_pairs_raw.csv")

df = pd.read_csv(raw_path)
print("Pairs in:", len(df))

df["findings"] = df["findings"].fillna("")
df["impression"] = df["impression"].fillna("")

if FINDINGS_ONLY:
    df = df[df["findings"].str.split().str.len() >= MIN_WORDS].copy()
    source = df["findings"]
    print("FINDINGS only")
else:
    source = (df["findings"] + " " + df["impression"]).str.strip()
    print("FINDINGS + IMPRESSION")

before = source.str.split().str.len()
df["report"] = source.apply(strip_temporal)
after = df["report"].str.split().str.len()

df = df[after >= MIN_WORDS].copy()
df["n_words"] = df["report"].str.split().str.len()

print(f"\nPairs out: {len(df)}  (dropped {len(pd.read_csv(raw_path)) - len(df)})")
print(f"Words before filtering : mean {before.mean():.1f}  median {before.median():.0f}")
print(f"Words after  filtering : mean {df['n_words'].mean():.1f}  median {df['n_words'].median():.0f}")
print(f"Reduction: {(1 - df['n_words'].mean() / before.mean()) * 100:.1f}%")

df.to_csv(cur_path, index=False)
print("\nSaved: image_report_pairs.csv (cleaned)")
