"""
Converts IN-CXR DICOM files to JPEG the way MIMIC-CXR-JPG was made, so the two
sets go through the same pipeline and any difference left is a real domain
difference rather than a conversion artefact.

Following Johnson et al. (MIMIC-CXR-JPG, arXiv:1901.07042), section 2:
invert when PhotometricInterpretation says to, rescale to 0-255 by subtracting
the minimum and dividing by the new maximum, apply histogram equalisation, and
write JPEG at quality 95. No resizing.

    python convert_dicom.py external/indian_data/intbtr251 --out external/in_cxr/images
    python convert_dicom.py external/indian_data/normal_1 --out external/in_cxr/images

Appends to external/in_cxr/manifest.csv, so several folders can be converted
one after another.
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from cxr_data import LABEL_COLS

EXTS = {".dicom", ".dcm", ".ima"}
NORMAL = "No Finding"
QUALITY = 95


def equalize_hist(a):
    """The mapping cv2.equalizeHist applies, without the dependency."""
    hist = np.bincount(a.ravel(), minlength=256)
    cdf = hist.cumsum()
    nz = cdf[cdf > 0]
    if nz.size == 0:
        return a
    cdf_min, total = nz[0], a.size
    if total == cdf_min:
        return a
    lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255).clip(0, 255).astype(np.uint8)
    return lut[a]


def to_jpeg_array(ds):
    a = ds.pixel_array.astype(np.float32)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        a = a.max() - a
    a = a - a.min()
    top = a.max()
    a = a / top if top > 0 else a
    a = (a * 255).astype(np.uint8)
    return equalize_hist(a)


def label_from_name(stem):
    s = stem.lower()
    if "abnormal" in s:
        return "abnormal"
    if "normal" in s:
        return "normal"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="external/in_cxr/images")
    ap.add_argument("--manifest", default="external/in_cxr/manifest.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print("not a folder:", args.root)
        return
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)

    files = []
    for dirpath, _, names in os.walk(args.root):
        for n in sorted(names):
            if os.path.splitext(n)[1].lower() in EXTS:
                files.append(os.path.join(dirpath, n))
    if args.limit:
        files = files[: args.limit]
    print("dicom files: %d" % len(files))
    if not files:
        return

    done = set()
    if os.path.isfile(args.manifest):
        done = set(pd.read_csv(args.manifest)["image_id"])
        print("already in manifest: %d" % len(done))

    rows, bad, photo, labels = [], 0, Counter(), Counter()
    for i, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in done:
            continue
        lab = label_from_name(stem)
        if lab is None:
            bad += 1
            continue
        try:
            ds = pydicom.dcmread(path)
            arr = to_jpeg_array(ds)
        except Exception as e:
            print("  unreadable: %s (%s)" % (stem[:28], type(e).__name__))
            bad += 1
            continue

        photo[ds.get("PhotometricInterpretation")] += 1
        labels[lab] += 1
        out_path = os.path.join(args.out, stem + ".jpg")
        Image.fromarray(arr).save(out_path, quality=QUALITY)

        row = {"image_id": stem,
               "local_path": out_path.replace("\\", "/"),
               "source": "in_cxr",
               "folder": os.path.basename(args.root.rstrip("/\\")),
               "split": "",
               "abnormal": 1 if lab == "abnormal" else 0,
               "width": arr.shape[1], "height": arr.shape[0], "mode": "L"}
        for c in LABEL_COLS:
            row[c] = 0
        if lab == "normal":
            row[NORMAL] = 1
        rows.append(row)

        if i % 100 == 0:
            print("  %d/%d" % (i, len(files)), flush=True)

    if not rows:
        print("nothing new to write")
        return

    df = pd.DataFrame(rows)
    if os.path.isfile(args.manifest):
        df = pd.concat([pd.read_csv(args.manifest), df], ignore_index=True)
    df.to_csv(args.manifest, index=False)

    print("\nconverted: %d   skipped: %d" % (len(rows), bad))
    print("photometric interpretation:", dict(photo))
    print("labels:", dict(labels))
    print("manifest now holds %d images (%d normal, %d abnormal)"
          % (len(df), int((df["abnormal"] == 0).sum()), int((df["abnormal"] == 1).sum())))
    print("saved", args.manifest)


if __name__ == "__main__":
    main()