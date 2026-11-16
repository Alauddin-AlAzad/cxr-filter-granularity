import csv
import os
import sys
import statistics as st
from collections import Counter, defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cxr_ratings_3.csv")
if not os.path.exists(PATH):
    sys.exit(f"ratings file not found: {PATH}\nusage: python analyse_ratings.py [path_to_ratings.csv]")
print(f"reading: {PATH}\n")

PATHOLOGICAL = [
    "pneumothorax", "effusion", "consolidation", "atelectasis", "opacit",
    "edema", "cardiomegaly", "enlarged", "lesion", "nodule", "mass",
    "fracture", "thickening", "hyperinflat", "congest", "infiltrat",
    "pneumonia", "emphysema", "indistinctness", "venous pressure",
]
TECHNICAL = ["low lung volume", "lung volumes are low"]
FINDING_TERMS = PATHOLOGICAL + TECHNICAL
NEGATORS = ["no ", "not ", "without", "free of", "negative for", "unremarkable",
            "are clear", "is clear", "within normal limits", "are intact",
            "grossly unremarkable", "no evidence"]
NORMALISERS = ["heart size is normal", "size is within normal limits",
               "cardiomediastinal silhouette is normal", "contours are normal",
               "vasculature is normal", "contours are unremarkable",
               "vasculature is not engorged", "heart size is top normal"]


def sentences(text):
    return [s.strip() for s in text.split(" . ") if s.strip()]


def positive_findings(text):
    """Return the sentences that assert a finding (not negated, not a normality statement)."""
    out = []
    for s in sentences(text.lower()):
        if any(n in s for n in NORMALISERS):
            continue
        hits = [t for t in FINDING_TERMS if t in s]
        if not hits:
            continue
        if any(neg in s for neg in NEGATORS):
            continue
        out.append((s, hits))
    return out


with open(PATH, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
assert "case" in rows[0], f"unexpected header: {list(rows[0].keys())}"

n = len(rows)
print(f"cases: {n}\n")

# ---------------------------------------------------------------- 1. scores
print("=== 1. SCORE DISTRIBUTIONS ===")
for col, label in [("correct_1_5", "Correctness"),
                   ("completeness_1_5", "Completeness"),
                   ("fluency_1_5", "Fluency")]:
    v = [int(r[col]) for r in rows]
    dist = Counter(v)
    print(f"{label:14s} mean {st.mean(v):.2f}  sd {st.stdev(v):.2f}  median {st.median(v):.1f}"
          f"  min {min(v)}  max {max(v)}")
    print(f"{'':14s} " + "  ".join(f"{k}:{dist.get(k,0)}" for k in range(1, 6))
          + f"   >=4: {sum(1 for x in v if x >= 4)}/{n} ({sum(1 for x in v if x >= 4)/n:.0%})")
print()

# ---------------------------------------------------------------- 2. reader read
print("=== 2. READER'S OWN READ ===")
reads = Counter(r["your_read"].strip() for r in rows)
print(dict(reads))
for rd in sorted(reads):
    sub = [int(r["correct_1_5"]) for r in rows if r["your_read"].strip() == rd]
    print(f"  {rd:10s} n={len(sub):2d}  correctness mean {st.mean(sub):.2f}")
print()

# ---------------------------------------------------------------- 3. template collapse
print("=== 3. DISTINCT GENERATED REPORTS (mode-collapse check) ===")
texts = [r["generated_report"].strip() for r in rows]
uniq = Counter(texts)
print(f"distinct reports: {len(uniq)} of {n}  ({len(uniq)/n:.1%})")
for t, c in uniq.most_common():
    if c > 1:
        print(f"  x{c:2d}  {t[:72]}...")
singles = sum(1 for c in uniq.values() if c == 1)
print(f"  reports appearing exactly once: {singles}")
print()

# ---------------------------------------------------------------- 4. findings named
print("=== 4. DOES THE REPORT NAME A FINDING? ===")
named, patho = {}, {}
for r in rows:
    pf = positive_findings(r["generated_report"])
    named[r["case"]] = pf
    terms = {t for _, hits in pf for t in hits}
    patho[r["case"]] = sorted(terms & set(PATHOLOGICAL))

with_f = [c for c, pf in named.items() if pf]
with_p = [c for c, t in patho.items() if t]
print(f"any finding term (incl. technical notes): {len(with_f)}/{n} ({len(with_f)/n:.1%})")
print(f"a PATHOLOGICAL finding:                   {len(with_p)}/{n} ({len(with_p)/n:.1%})")
print("pathological cases: " + ", ".join(with_p))
print()
for c in with_f:
    terms = sorted({t for _, hits in named[c] for t in hits})
    tag = "PATH" if patho[c] else "tech only"
    print(f"  case {c:>2s} [{tag:9s}]: {', '.join(terms)}")
print()

print("--- cross-tab: reader's read x report names a PATHOLOGICAL finding ---")
ct = Counter()
for r in rows:
    ct[(r["your_read"].strip(), bool(patho[r["case"]]))] += 1
for k in sorted(ct):
    print(f"  read={k[0]:9s} report_names_pathology={str(k[1]):5s} : {ct[k]}")
ab = [r for r in rows if r["your_read"].strip().lower() == "abnormal"]
nm = [r for r in rows if r["your_read"].strip().lower() == "normal"]
ab_named = [r for r in ab if patho[r["case"]]]
nm_named = [r for r in nm if patho[r["case"]]]
print(f"\n  reader called ABNORMAL: {len(ab)}; report named a pathological finding in "
      f"{len(ab_named)} ({len(ab_named)/len(ab):.1%})")
print("    abnormal cases where the report named no pathology: "
      + ", ".join(r["case"] for r in ab if not patho[r["case"]]))
print(f"  reader called NORMAL: {len(nm)}; report named a pathological finding in "
      f"{len(nm_named)} ({len(nm_named)/len(nm):.1%}) -> possible false positives: "
      + ", ".join(r["case"] for r in nm_named))
print()
print(f"  NOTE: the sample was built 25 normal / 25 abnormal by the dataset's reference label,")
print(f"  but the reader called only {len(ab)} abnormal. That gap is itself a result.")
print()

# ---------------------------------------------------------------- 5. length / truncation
print("=== 5. REPORT LENGTH AND TRUNCATION ===")
lens = [len(t.split()) for t in texts]
print(f"words per report: mean {st.mean(lens):.1f}  median {st.median(lens):.0f}  "
      f"min {min(lens)}  max {max(lens)}")
trunc = [r["case"] for r in rows if not r["generated_report"].strip().endswith(".")]
print(f"reports not ending in a full stop (possible truncation): {trunc}")
for c in trunc:
    t = [r["generated_report"] for r in rows if r["case"] == c][0]
    print(f"  case {c}: ...{t[-70:]}")
    comp = [r["completeness_1_5"] for r in rows if r["case"] == c][0]
    print(f"           completeness scored: {comp}")
print()

# ---------------------------------------------------------------- 6. would_sign
print("=== 6. WOULD_SIGN ===")
print(dict(Counter(r["would_sign"].strip() for r in rows)))
low = [(r["case"], r["correct_1_5"], r["would_sign"].strip()) for r in rows
       if int(r["correct_1_5"]) <= 2]
print(f"cases with correctness <= 2: {len(low)}")
for c, cor, ws in low:
    print(f"  case {c:>2s}  correctness {cor}  would_sign {ws}")
print()

# ---------------------------------------------------------------- 7. free text
print("=== 7. MISSED_OR_WRONG FREE TEXT ===")
blank, no_, text_ = [], [], []
for r in rows:
    v = r["missed_or_wrong"].strip()
    if v == "":
        blank.append(r["case"])
    elif v.lower() == "no":
        no_.append(r["case"])
    else:
        text_.append((r["case"], " ".join(v.split())))
print(f"blank: {blank}")
print(f"'No': {len(no_)} cases")
print(f"with text: {len(text_)}")
for c, v in text_:
    print(f"  case {c:>2s}: {v}")
phr = Counter(v for _, v in text_)
print("\nrepeated verbatim entries:")
for v, c in phr.most_common():
    if c > 1:
        print(f"  x{c}  {v}")
print()

# ---------------------------------------------------------------- 8. agreement between scores
print("=== 8. RELATION BETWEEN THE THREE SCORES ===")


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


cor = [int(r["correct_1_5"]) for r in rows]
com = [int(r["completeness_1_5"]) for r in rows]
flu = [int(r["fluency_1_5"]) for r in rows]
print(f"correctness vs completeness: r = {pearson(cor, com):+.3f}")
print(f"correctness vs fluency:      r = {pearson(cor, flu):+.3f}")
print(f"completeness vs fluency:     r = {pearson(com, flu):+.3f}")
print()

# ------------------------------------------------- 9. same text, different score
print("=== 9. IDENTICAL TEXT, DIFFERENT CORRECTNESS ===")
by_text = defaultdict(list)
for r in rows:
    by_text[r["generated_report"].strip()].append(r)
for t, grp in sorted(by_text.items(), key=lambda kv: -len(kv[1])):
    if len(grp) < 2:
        continue
    sc = [int(g["correct_1_5"]) for g in grp]
    reads = Counter(g["your_read"].strip() for g in grp)
    print(f"  n={len(grp):2d}  correctness {min(sc)}-{max(sc)} (mean {st.mean(sc):.2f})  "
          f"reads {dict(reads)}")
    print(f"        {t[:70]}...")
same = [g for grp in by_text.values() if len(grp) > 1 for g in grp]
print(f"\n  cases sharing their text with another case: {len(same)}/{n} ({len(same)/n:.0%})")
sc_all = [int(g["correct_1_5"]) for g in same]
print(f"  their correctness ranges {min(sc_all)}-{max(sc_all)}, mean {st.mean(sc_all):.2f}")