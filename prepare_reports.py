"""
Extracts FINDINGS / IMPRESSION from the MIMIC-CXR free-text reports and pairs
each study with the images already selected in subset_labels.csv.

Input : mimic-cxr-reports.zip, subset_labels.csv
Output: study_reports.csv, image_report_pairs.csv
"""

import re
import zipfile

import pandas as pd

DATA_DIR = "."
REPORT_ZIP = "mimic-cxr-reports.zip"
MIN_WORDS = 5

SECTION_RE = re.compile(
    r"^\s*(FINAL REPORT|EXAMINATION|INDICATION|HISTORY|TECHNIQUE|COMPARISON|"
    r"FINDINGS|IMPRESSION|CONCLUSION|RECOMMENDATION\(S\)|NOTIFICATION|WET READ)\s*:",
    re.MULTILINE,
)


def split_sections(text):
    marks = [(m.start(), m.group(1).upper()) for m in SECTION_RE.finditer(text)]
    out = {}
    for k, (pos, name) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        body = text[text.index(":", pos) + 1:end]
        out[name] = " ".join(body.split())
    return out


def clean(s):
    s = re.sub(r"_{2,}", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


labels = pd.read_csv(f"{DATA_DIR}/subset_labels.csv")
wanted = set(labels["study_id"].astype(int))
print("Studies needed:", len(wanted))

rows = []
missing_findings = 0

with zipfile.ZipFile(f"{DATA_DIR}/{REPORT_ZIP}") as z:
    names = [n for n in z.namelist() if n.endswith(".txt")]
    print("Report files in archive:", len(names))

    for n in names:
        stem = n.rsplit("/", 1)[-1]
        if not stem.startswith("s"):
            continue
        try:
            sid = int(stem[1:-4])
        except ValueError:
            continue
        if sid not in wanted:
            continue

        text = z.read(n).decode("utf-8", errors="ignore")
        sec = split_sections(text)

        findings = clean(sec.get("FINDINGS", ""))
        impression = clean(sec.get("IMPRESSION", sec.get("CONCLUSION", "")))

        if not findings:
            missing_findings += 1

        report = " ".join(x for x in [findings, impression] if x)
        if len(report.split()) < MIN_WORDS:
            continue

        rows.append({
            "study_id": sid,
            "findings": findings,
            "impression": impression,
            "report": report,
            "n_words": len(report.split()),
        })

reports = pd.DataFrame(rows).drop_duplicates(subset=["study_id"])
reports.to_csv(f"{DATA_DIR}/study_reports.csv", index=False)

print(f"\nStudies with usable text : {len(reports)}")
print(f"Missing FINDINGS section : {missing_findings}")
print(f"Report length (words)    : mean {reports['n_words'].mean():.1f}  "
      f"median {reports['n_words'].median():.0f}  max {reports['n_words'].max()}")

pairs = labels.merge(reports[["study_id", "findings", "impression", "report", "n_words"]],
                     on="study_id", how="inner")
pairs.to_csv(f"{DATA_DIR}/image_report_pairs.csv", index=False)

print(f"\nImage-report pairs       : {len(pairs)}")
print(f"Images with no report    : {len(labels) - len(pairs)}")
print("\nSaved: study_reports.csv, image_report_pairs.csv")