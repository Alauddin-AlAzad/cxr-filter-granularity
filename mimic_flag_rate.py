"""
MIMIC-CXR: on abnormal TEST images, how often does the classifier flag
at least one finding?  (In-domain partner of the IN-CXR figure 72.2%.)

No GPU, no training. Reads files you already have in E:\\mimic_data.
Prints only aggregate numbers - safe to paste in chat (no images, no reports).

Run:
    conda activate cxr
    cd /d E:\\mimic_data
    python mimic_flag_rate.py
Optional:
    python mimic_flag_rate.py --thresholds prompt_thresholds_calibrated.json
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "Pleural Effusion", "Pleural Other",
            "Pneumonia", "Pneumothorax"]          # 12 findings, No Finding excluded
KEY_CANDIDATES = ["dicom_id", "path", "image_path", "image", "filename", "file"]
LABEL_FILES = ["image_report_pairs.csv", "available_labels.csv",
               "subset_labels.csv", "mimic-cxr-2.0.0-chexpert.csv",
               "mimic-cxr-2.1.0-chexpert.csv"]

ap = argparse.ArgumentParser()
ap.add_argument("--probs", default="prompt_probs.csv")
ap.add_argument("--thresholds", default="prompt_thresholds.json")
ap.add_argument("--labels", default=None, help="label CSV (auto-detected if omitted)")
args = ap.parse_args()

def stop(msg):
    print("\nSTOP:", msg); sys.exit(1)

def find_col(df, name):
    """exact name, or a column that contains it (e.g. 'prob_Edema')."""
    if name in df.columns: return name
    hits = [c for c in df.columns if name.lower() in c.lower()]
    return hits[0] if len(hits) == 1 else None

def norm_key(s):
    """compare images by file stem, so 'files/p10/.../abc.jpg' == 'abc'."""
    return s.astype(str).str.replace("\\", "/").str.split("/").str[-1].str.replace(".jpg", "", regex=False)

# ---- probabilities ---------------------------------------------------------
if not os.path.exists(args.probs): stop(f"{args.probs} not found in {os.getcwd()}")
probs = pd.read_csv(args.probs)
pkey = next((k for k in KEY_CANDIDATES if k in probs.columns), None)
pcols = {f: find_col(probs, f) for f in FINDINGS}
print("probs file      :", args.probs, probs.shape)
print("probs key column:", pkey)
missing = [f for f, c in pcols.items() if c is None]
if pkey is None or missing:
    print("columns found   :", list(probs.columns)[:40])
    stop("could not match key or finding columns - paste the 'columns found' line to Claude")

# ---- thresholds ------------------------------------------------------------
th_raw = json.load(open(args.thresholds))
if isinstance(th_raw, dict) and "thresholds" in th_raw: th_raw = th_raw["thresholds"]
thr = {}
for f in FINDINGS:
    v = th_raw.get(f) if isinstance(th_raw, dict) else None
    if isinstance(v, dict): v = v.get("threshold")
    thr[f] = None if v is None else float(v)
never = [f for f, v in thr.items() if v is None]
print("thresholds      :", args.thresholds, "| never prompted:", never or "none")

# ---- labels ----------------------------------------------------------------
lab_path = args.labels or next((p for p in LABEL_FILES if os.path.exists(p)), None)
if lab_path is None: stop("no label CSV found; pass --labels <file>")
labels = pd.read_csv(lab_path)
lkey = next((k for k in KEY_CANDIDATES if k in labels.columns), None)
lcols = {f: find_col(labels, f) for f in FINDINGS}
print("labels file     :", lab_path, labels.shape, "| key:", lkey)
if lkey is None or any(c is None for c in lcols.values()):
    print("columns found   :", list(labels.columns)[:40])
    stop("could not match label columns - paste the 'columns found' line to Claude")

probs["_k"] = norm_key(probs[pkey]); labels["_k"] = norm_key(labels[lkey])
labels = labels.drop_duplicates("_k")
df = probs.merge(labels[["_k"] + list(lcols.values()) +
                        (["split"] if "split" in labels.columns and "split" not in probs.columns else [])],
                 on="_k", how="inner", suffixes=("_p", ""))
if "split" in df.columns:
    df = df[df["split"].astype(str).str.lower() == "test"]
    print("split           : test only")
else:
    print("split           : NO split column found - using every matched image (check this!)")
print("matched images  :", len(df))
if len(df) == 0: stop("no images matched between the two files")

# ---- the measure -----------------------------------------------------------
# label 1 = positive; 0, -1 (uncertain) and blank = not positive
pos = np.column_stack([(df[lcols[f]] == 1).to_numpy() for f in FINDINGS])
flag = np.column_stack([
    (df[pcols[f]].to_numpy() >= thr[f]) if thr[f] is not None else np.zeros(len(df), bool)
    for f in FINDINGS])
abnormal = pos.any(1)
flagged = flag.any(1)

def rate_ci(x, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return x.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

print("\n================ RESULT ================")
m, lo, hi = rate_ci(flagged[abnormal])
print(f"abnormal test images             : {abnormal.sum()}")
print(f"classifier flags >=1 finding     : {m*100:.1f}%  95% CI [{lo*100:.1f}%, {hi*100:.1f}%]")
m2, lo2, hi2 = rate_ci(flagged[~abnormal])
print(f"normal test images               : {(~abnormal).sum()}")
print(f"  flagged anyway (false alarm)   : {m2*100:.1f}%  95% CI [{lo2*100:.1f}%, {hi2*100:.1f}%]")
print("compare with IN-CXR abnormal     : 72.2%")
print("========================================")