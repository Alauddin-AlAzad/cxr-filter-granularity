"""
Reports the pixel size of the images embedded inside the Word reports.

A .docx is a zip; the pictures sit in word/media/. If those are larger than
the exported BMP files, the full-resolution radiographs are recoverable from
the documents. If they are the same size, the originals must come from the
clinic.

    python check_docx_images.py E:/bd_reports_private
"""

import io
import os
import sys
import zipfile
from collections import Counter

from PIL import Image


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "E:/bd_reports_private"
    docs = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".docx"):
                docs.append(os.path.join(dirpath, f))

    print("documents found: %d" % len(docs))
    if not docs:
        print("no .docx under", root)
        return

    sizes, kinds, counts, failed = Counter(), Counter(), Counter(), 0
    for path in docs[:60]:
        try:
            with zipfile.ZipFile(path) as z:
                media = [n for n in z.namelist() if n.startswith("word/media/")]
                counts[len(media)] += 1
                for n in media:
                    data = z.read(n)
                    kinds[os.path.splitext(n)[1].lower()] += 1
                    try:
                        with Image.open(io.BytesIO(data)) as im:
                            sizes[im.size] += 1
                    except Exception:
                        failed += 1
        except Exception as e:
            print("  could not open %s (%s)" % (os.path.basename(path), type(e).__name__))

    print("inspected: %d documents" % min(len(docs), 60))
    print("images per document:", dict(counts))
    print("file types:", dict(kinds))
    print("unreadable: %d" % failed)
    print("\nembedded image sizes:")
    for size, n in sizes.most_common(10):
        print("  %-14s %4d images" % ("%dx%d" % size, n))


if __name__ == "__main__":
    main()