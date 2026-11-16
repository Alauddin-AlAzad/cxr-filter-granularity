"""
Reports what is inside the IN-CXR DICOM files.

    python inspect_dicom.py <folder>

Prints the image size, bit depth and windowing tags, says whether patient
identifiers are still in the file, and saves one preview PNG to look at.
"""

import os
import sys
from collections import Counter

import numpy as np
import pydicom
from PIL import Image

EXTS = {".dicom", ".dcm", ".ima"}

ID_TAGS = ["PatientName", "PatientID", "PatientBirthDate", "PatientAge",
           "PatientSex", "InstitutionName", "StudyDate", "AccessionNumber",
           "Manufacturer", "ManufacturerModelName"]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(root):
        print("not a folder:", root)
        return

    files, other = [], Counter()
    for dirpath, _, names in os.walk(root):
        for n in names:
            ext = os.path.splitext(n)[1].lower()
            if ext in EXTS:
                files.append(os.path.join(dirpath, n))
            else:
                other[ext or "(no extension)"] += 1

    print("dicom files found: %d" % len(files))
    if other:
        print("other files:", dict(other))
    if not files:
        return

    labels = Counter()
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        labels["abnormal" if "abnormal" in stem else
               "normal" if "normal" in stem else "unlabelled"] += 1
    print("labels from file names:", dict(labels))

    sizes, depths, photo = Counter(), Counter(), Counter()
    for path in files[:60]:
        try:
            d = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as e:
            print("  unreadable: %s (%s)" % (os.path.basename(path), type(e).__name__))
            continue
        sizes[(d.get("Rows"), d.get("Columns"))] += 1
        depths[d.get("BitsStored")] += 1
        photo[d.get("PhotometricInterpretation")] += 1

    print("\nimage sizes (sample of 60):", dict(sizes))
    print("bits stored:", dict(depths))
    print("photometric interpretation:", dict(photo))

    d = pydicom.dcmread(files[0])
    print("\ntags on the first file:")
    for t in ID_TAGS:
        v = d.get(t, None)
        print("  %-24s %s" % (t, "(empty)" if v in (None, "") else str(v)[:46]))
    print("  %-24s %s" % ("WindowCenter", d.get("WindowCenter")))
    print("  %-24s %s" % ("WindowWidth", d.get("WindowWidth")))
    print("  %-24s %s" % ("RescaleSlope", d.get("RescaleSlope")))
    print("  %-24s %s" % ("RescaleIntercept", d.get("RescaleIntercept")))

    a = d.pixel_array.astype(np.float32)
    print("\npixel range: %.0f to %.0f   shape %s   dtype %s"
          % (a.min(), a.max(), a.shape, d.pixel_array.dtype))

    a = (a - a.min()) / max(a.max() - a.min(), 1)
    if d.get("PhotometricInterpretation") == "MONOCHROME1":
        a = 1.0 - a
        print("MONOCHROME1: inverted for the preview")
    im = Image.fromarray((a * 255).astype(np.uint8))
    im.thumbnail((900, 900))
    im.save("dicom_preview.png")
    print("saved dicom_preview.png")


if __name__ == "__main__":
    main()