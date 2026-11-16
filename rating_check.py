import csv
import os
import sys
from collections import Counter

NORMAL_WORDS = {"0", "0.0", "normal", "no finding", "no_finding", "nofinding",
                "false", "no", "n", "negative", "-"}
ABNORMAL_WORDS = {"1", "1.0", "abnormal", "true", "yes", "y", "positive", "+"}


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def stem(s):
    return os.path.splitext(os.path.basename(s.strip()))[0].lower()


def find_image_col(key_rows, wanted):
    best, best_hits = None, 0
    for col in key_rows[0].keys():
        hits = sum(1 for r in key_rows if stem(str(r[col] or "")) in wanted)
        if hits > best_hits:
            best, best_hits = col, hits
    return best, best_hits


def classify(value):
    v = str(value).strip().lower()
    if v in NORMAL_WORDS:
        return "normal"
    if v in ABNORMAL_WORDS:
        return "abnormal"
    return None


def find_label_col(key_rows, image_col):
    cands = []
    for col in key_rows[0].keys():
        if col == image_col:
            continue
        vals = [str(r[col] or "").strip() for r in key_rows]
        mapped = [classify(v) for v in vals]
        if all(m is not None for m in mapped) and len(set(mapped)) == 2:
            cands.append((col, Counter(mapped)))
    return cands


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python rating_check.py <ratings.csv> <key.csv>")
    ratings_path, key_path = sys.argv[1], sys.argv[2]
    for p in (ratings_path, key_path):
        if not os.path.exists(p):
            sys.exit(f"file not found: {p}")

    ratings = load(ratings_path)
    key_rows = load(key_path)
    print(f"ratings rows: {len(ratings)}   key rows: {len(key_rows)}")
    print(f"key columns: {list(key_rows[0].keys())}\n")

    wanted = {stem(r["image"]) for r in ratings}
    image_col, hits = find_image_col(key_rows, wanted)
    join = "image"
    if not image_col or hits == 0:
        case_col = next((c for c in key_rows[0] if c.strip().lower() == "case"), None)
        if case_col:
            keycases = {str(r[case_col]).strip() for r in key_rows}
            if sum(1 for r in ratings if r["case"].strip() in keycases) > 0:
                image_col, join = case_col, "case"
                hits = sum(1 for r in ratings if r["case"].strip() in keycases)
                print(f"image names did not match; joining on the 'case' column instead")
    if not image_col or hits == 0:
        print("Could not match the ratings to the key file on image name or case number.")
        print(f"ratings image examples: {sorted(wanted)[:3]}")
        print("key file first row:")
        for k, v in key_rows[0].items():
            print(f"   {k} = {v}")
        sys.exit(1)
    print(f"joining on '{image_col}' ({join})  -  matches {hits} of {len(ratings)} rated cases")

    cands = find_label_col(key_rows, image_col)
    if not cands:
        print("\nNo clean two-value normal/abnormal column found. Column value counts:")
        for col in key_rows[0].keys():
            if col == image_col:
                continue
            c = Counter(str(r[col] or "").strip() for r in key_rows)
            if len(c) <= 8:
                print(f"   {col}: {dict(c)}")
        sys.exit(1)
    label_col, counts = cands[0]
    if len(cands) > 1:
        print(f"candidate label columns: {[c[0] for c in cands]} -> using '{label_col}'")
    print(f"label column in key: '{label_col}'   values: {dict(counts)}\n")

    if join == "case":
        key = {str(r[image_col]).strip(): classify(r[label_col]) for r in key_rows}
    else:
        key = {stem(str(r[image_col])): classify(r[label_col]) for r in key_rows}

    rows, missing = [], []
    for r in ratings:
        img = r["case"].strip() if join == "case" else stem(r["image"])
        if img not in key:
            missing.append(r["case"])
            continue
        rows.append({
            "case": r["case"],
            "read": "normal" if r["your_read"].strip().lower() == "normal" else "abnormal",
            "key": key[img],
            "correct": int(r["correct_1_5"]),
        })

    n = len(rows)
    if missing:
        print(f"no key entry for cases: {missing}")
    print(f"=== MATCHED: {n} of {len(ratings)} cases ===\n")
    if n == 0:
        sys.exit("nothing to compare")

    print("key label balance in the rated sample:",
          dict(Counter(x["key"] for x in rows)))
    print("reader's own reads:            ",
          dict(Counter(x["read"] for x in rows)))
    print()

    agree = sum(1 for x in rows if x["read"] == x["key"])
    print(f"AGREEMENT (reader vs key): {agree}/{n} = {agree / n:.1%}\n")

    print("confusion matrix")
    print(f"{'':22s}{'key=normal':>14s}{'key=abnormal':>14s}")
    for rd in ("normal", "abnormal"):
        a = sum(1 for x in rows if x["read"] == rd and x["key"] == "normal")
        b = sum(1 for x in rows if x["read"] == rd and x["key"] == "abnormal")
        print(f"reader={rd:<16s}{a:>14d}{b:>14d}")
    print()

    tp = sum(1 for x in rows if x["read"] == "abnormal" and x["key"] == "abnormal")
    fn = sum(1 for x in rows if x["read"] == "normal" and x["key"] == "abnormal")
    fp = sum(1 for x in rows if x["read"] == "abnormal" and x["key"] == "normal")
    tn = sum(1 for x in rows if x["read"] == "normal" and x["key"] == "normal")
    if tp + fn:
        print(f"reader sensitivity (abnormal caught): {tp}/{tp + fn} = {tp / (tp + fn):.1%}")
    if tn + fp:
        print(f"reader specificity (normal kept):     {tn}/{tn + fp} = {tn / (tn + fp):.1%}")
    print()

    dis = [x for x in rows if x["read"] != x["key"]]
    print(f"disagreements: {len(dis)}")
    print("  reader said NORMAL, key says abnormal: "
          + ", ".join(x["case"] for x in dis if x["read"] == "normal"))
    print("  reader said ABNORMAL, key says normal: "
          + ", ".join(x["case"] for x in dis if x["read"] == "abnormal"))
    print()

    for c in ("25", "28"):
        m = [x for x in rows if x["case"] == c]
        if m:
            x = m[0]
            print(f"case {c}: key={x['key']}  reader={x['read']}  correctness={x['correct']}")
    print()

    ok = [x["correct"] for x in rows if x["read"] == x["key"]]
    bad = [x["correct"] for x in rows if x["read"] != x["key"]]
    if ok:
        print(f"mean correctness where reader agrees with key:    {sum(ok) / len(ok):.2f} (n={len(ok)})")
    if bad:
        print(f"mean correctness where reader differs from key:   {sum(bad) / len(bad):.2f} (n={len(bad)})")
    print()

    for grp in ("normal", "abnormal"):
        sub = [x["correct"] for x in rows if x["key"] == grp]
        if sub:
            print(f"mean correctness on key-{grp:8s} images: {sum(sub) / len(sub):.2f} (n={len(sub)})")


main()