"""
Builds a manifest CSV for an external image folder, in the same label layout
as available_labels.csv, so the existing evaluation code can read it.

    python make_external_manifest.py external/bd_clinic --all-normal --out external/bd_clinic.csv
    python make_external_manifest.py external/bd_online --label-from-folder --out external/bd_online.csv
    python make_external_manifest.py external/bd_online --label-from-name --out external/bd_online.csv

Every image is opened once, so truncated files are caught here rather than
during evaluation. Width and height are recorded for the resolution-matched
control.
"""

import argparse
import os
from collections import Counter

import pandas as pd
from PIL import Image

from cxr_data import LABEL_COLS

IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
NORMAL = "No Finding"


def label_for_text(name):
    """Reads a label out of a folder or file name."""
    n = name.lower()
    if "normal" in n or "healthy" in n:
        return NORMAL
    for col in LABEL_COLS:
        if col == NORMAL:
            continue
        if col.lower().split()[0] in n:
            return col
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--all-normal", action="store_true")
    ap.add_argument("--label-from-folder", action="store_true")
    ap.add_argument("--label-from-name", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("label columns:", LABEL_COLS)
    if NORMAL not in LABEL_COLS:
        print("warning: '%s' is not a label column" % NORMAL)

    rows, bad, folders, unlabelled = [], 0, Counter(), Counter()
    for dirpath, _, files in os.walk(args.root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in IMG_EXTS:
                continue
            path = os.path.join(dirpath, f)
            try:
                with Image.open(path) as im:
                    w, h, mode = im.width, im.height, im.mode
                    im.convert("RGB")
            except Exception as e:
                bad += 1
                print("  unreadable: %s (%s)" % (f, type(e).__name__))
                continue

            folder = os.path.basename(dirpath)
            parent = os.path.basename(os.path.dirname(dirpath)).lower()
            folders[folder] += 1
            row = {"image_id": os.path.splitext(f)[0],
                   "local_path": path.replace("\\", "/"),
                   "source": os.path.basename(args.root.rstrip("/\\")),
                   "folder": folder,
                   "split": parent if parent in ("train", "test", "val", "validate") else "",
                   "width": w, "height": h, "mode": mode}
            for c in LABEL_COLS:
                row[c] = 0

            if args.all_normal:
                row[NORMAL] = 1
            else:
                source = row["image_id"] if args.label_from_name else folder
                lab = label_for_text(source)
                if lab is None:
                    unlabelled[source if args.label_from_name else folder] += 1
                    continue
                row[lab] = 1
            rows.append(row)

    if not rows:
        print("no usable images found")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\nimages written: %d   unreadable: %d" % (len(df), bad))
    print("folders:", dict(folders))
    if unlabelled:
        total = sum(unlabelled.values())
        print("skipped, no label found: %d" % total)
        for name, n in unlabelled.most_common(10):
            print("  %-40s %5d" % (name[:40], n))
    if args.label_from_folder or args.label_from_name:
        print("label counts:")
        for c in LABEL_COLS:
            if df[c].sum():
                print("  %-28s %5d" % (c, int(df[c].sum())))
        if df["split"].any():
            print("by split:")
            for s, g in df.groupby("split"):
                hits = ", ".join("%s %d" % (c, int(g[c].sum())) for c in LABEL_COLS if g[c].sum())
                print("  %-10s %5d   (%s)" % (s or "unsplit", len(g), hits))
    print("\nimage size: width %d-%d   height %d-%d"
          % (df.width.min(), df.width.max(), df.height.min(), df.height.max()))
    print("most common sizes:", Counter(zip(df.width, df.height)).most_common(5))
    print("colour modes:", dict(Counter(df["mode"])))
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()