"""
Removes comparisons with a prior study while keeping the finding.

The sentence-level filter in clean_reports.py deletes a whole sentence when it
contains a temporal word, so "Persistent bilateral pleural effusions" is lost
along with the comparison. This removes the comparative phrase instead and
keeps the finding, except where removing it would reverse the meaning.

    python clean_reports_phrase.py

Reads image_report_pairs_raw.csv, writes image_report_pairs_phrase.csv.
Nothing else is overwritten.
"""

import os
import re

import pandas as pd

DATA_DIR = "."
RAW = "image_report_pairs_raw.csv"
OUT = "image_report_pairs_phrase.csv"
MIN_WORDS = 4

# removing these would assert a finding that is no longer there, or the
# sentence is only about change, so the sentence goes
DROP_SENTENCE = [
    "resolved", "resolving", "resolution of", "no longer", "has been removed",
    "have been removed", "has been", "have been", "removed", "cleared",
    "no change", "no significant change", "little change", "no interval change",
    "interval change", "has decreased", "no residual",
]

# a comparison clause: delete from the phrase to the next comma or the end
CLAUSE = [
    "compared to", "compared with", "as compared to", "as compared with",
    "when compared to", "when compared with", "in comparison to",
    "in comparison with", "in comparison", "comparison is made",
    "since the prior", "since the previous", "since prior", "since the",
    "relative to the prior", "relative to the previous", "similar to the prior",
    "similar to the previous", "as before", "as on the prior",
    "from the prior", "from the previous", "from prior", "from previous",
    "on the prior", "on the previous", "than the prior", "than prior",
    "versus the prior", "versus prior",
]

# rewritten before anything is removed, so the sentence still reads
PHRASE_FIX = [
    (r"\bagain seen (is|are)\b", r"there \1"),
    (r"\bagain noted\b", "noted"),
    (r"\bagain demonstrated\b", "demonstrated"),
    (r"\bcontinues? to be\b", "is"),
    (r"\bremains? \b", "is "),
]

# a fragment made only of these carries no finding
STOP = {"is", "are", "was", "were", "the", "a", "an", "of", "and", "with",
        "in", "on", "to", "from", "there", "this", "that", "no", "not",
        "seen", "noted", "demonstrated", "likely", "which", "also", "it"}

# qualifiers that can be dropped with the finding left standing
STRIP_WORDS = [
    "persistently", "persistent", "persisting", "persists", "persist",
    "unchanged", "stable", "constant", "again", "redemonstrated",
    "re-demonstrated", "redemonstration of", "newly", "new",
    "increased", "increasing", "decreased", "decreasing",
    "improved", "improving", "worsened", "worsening", "worsened",
    "progressed", "progression of", "progressive", "continued", "continues",
    "continue to be", "ongoing", "status post", "postoperative",
    "previously", "previous", "prior", "earlier", "interval",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DROP_RE = re.compile("|".join(re.escape(t) for t in DROP_SENTENCE))
_CLAUSE_RE = re.compile("|".join(r"\b%s\b[^.,;]*[,;]?" % re.escape(t) for t in CLAUSE))
_STRIP_RE = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(t) for t in
                                               sorted(STRIP_WORDS, key=len, reverse=True)))
_WORD = re.compile(r"[a-z]+")


def tidy(s):
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(from|to|with|since|than|versus|vs|as)\s*(?=[,.;:]|$)", "", s)
    s = re.sub(r"\s+([.,;:])", r"\1", s)
    s = re.sub(r"(?:\s*,)+\s*(?=[.,;:])", "", s)
    s = re.sub(r",\s*\.", ".", s)
    s = re.sub(r"\b(is|are|was|were|has|have)\s*(?=[.,;:]|$)", "", s)
    s = re.sub(r"\bthere (is|are)\s*(?=[.,;:]|$)", "", s)
    s = re.sub(r"^[\s,;:]+", "", s)
    s = re.sub(r"^(is|are|of|and|with|the|from|to)\b\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([.,;:])", r"\1", s)
    return s.strip(" ,;:")


def clean_sentence(s):
    low = s.lower()
    if _DROP_RE.search(low):
        return ""
    for pat, rep in PHRASE_FIX:
        low = re.sub(pat, rep, low)
    out = _CLAUSE_RE.sub(" ", low)
    out = _STRIP_RE.sub(" ", out)
    out = tidy(out)
    words = _WORD.findall(out)
    if not words or all(w in STOP for w in words):
        return ""
    return out


def clean_report(text):
    kept = [clean_sentence(s) for s in _SENT_SPLIT.split(str(text)) if s.strip()]
    kept = [k for k in kept if k]
    out = " ".join(kept)
    if out and not out.endswith("."):
        out += "."
    return out


def main():
    raw_path = os.path.join(DATA_DIR, RAW)
    if not os.path.isfile(raw_path):
        print("not found:", raw_path)
        return

    df = pd.read_csv(raw_path)
    print("pairs in:", len(df))

    df["findings"] = df["findings"].fillna("")
    df["impression"] = df["impression"].fillna("")
    source = (df["findings"] + " " + df["impression"]).str.strip()

    before = source.str.split().str.len()
    df["report"] = source.apply(clean_report)
    after = df["report"].str.split().str.len()

    df = df[after >= MIN_WORDS].copy()
    df["n_words"] = df["report"].str.split().str.len()

    print("pairs out: %d  (dropped %d)" % (len(df), len(source) - len(df)))
    print("words before: mean %.1f  median %.0f" % (before.mean(), before.median()))
    print("words after : mean %.1f  median %.0f" % (df["n_words"].mean(), df["n_words"].median()))
    print("reduction: %.1f%%" % ((1 - df["n_words"].mean() / before.mean()) * 100))

    leftover = df["report"].str.contains(
        r"\b(compared|comparison|previous|prior|unchanged|persistent|interval)\b",
        case=False, regex=True).mean()
    print("reports still containing a comparison word: %.2f%%" % (100 * leftover))

    df.to_csv(os.path.join(DATA_DIR, OUT), index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()