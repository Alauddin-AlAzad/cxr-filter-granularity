"""
False-positive rate of the classifier on normal chest radiographs, measured on
an external set and on MIMIC, so the two can be compared.

A false-positive rate on its own says nothing: the model may call findings on
MIMIC normals just as often. The MIMIC arm is the baseline, and the third arm
repeats it with MIMIC images shrunk to the external set's resolution, which
separates a domain effect from a resolution effect.

    python external_eval.py --manifest external/bd_clinic.csv
    python external_eval.py --manifest external/bd_clinic.csv --match_size 1024

Writes external_eval.json.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cxr_data import CXRDataset, LABEL_COLS, NORM, PadSquare, load_frames
from train_classifier import build_model

CKPT = "checkpoints_320/best.pt"
THRESHOLDS = "prompt_thresholds.json"
IMG_SIZE = 320
BATCH = 16
N_BOOT = 1000
SEED = 0
NORMAL = "No Finding"
DISEASES = [c for c in LABEL_COLS if c != NORMAL]


def eval_transform(img_size=IMG_SIZE, match_size=None):
    """match_size shrinks the long side first, mimicking a lower-resolution
    source before the usual padding and resize."""
    steps = []
    if match_size:
        steps.append(transforms.Resize(match_size))
    steps += [PadSquare(), transforms.Resize((img_size, img_size)), transforms.ToTensor(), NORM]
    return transforms.Compose(steps)


def load_thresholds():
    if not os.path.isfile(THRESHOLDS):
        print("no %s, using 0.5 for every class" % THRESHOLDS)
        return {c: 0.5 for c in DISEASES}
    th = json.load(open(THRESHOLDS))
    out = {}
    for c in DISEASES:
        t = th.get(c, 0.5)
        out[c] = float(t) if isinstance(t, (int, float)) and 0 < t <= 1 else 0.5
    return out


@torch.no_grad()
def predict(model, df, tf, device):
    loader = DataLoader(CXRDataset(df, tf), batch_size=BATCH, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            logit = model(x.to(device))
        out.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(out)


def flags(probs, th):
    """Boolean matrix of which diseases each image was called positive for."""
    idx = [LABEL_COLS.index(c) for c in DISEASES]
    cut = np.array([th[c] for c in DISEASES])
    return probs[:, idx] >= cut


def rates(fl, rng):
    n = len(fl)
    any_fp = fl.any(axis=1)
    per_image = fl.sum(axis=1)

    boot = [any_fp[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {"n": int(n),
            "images_with_any_false_positive": round(float(any_fp.mean()), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "mean_findings_per_image": round(float(per_image.mean()), 3),
            "per_class_false_positive_rate":
                {c: round(float(fl[:, j].mean()), 4) for j, c in enumerate(DISEASES)}}


def report(name, r):
    print("\n%s  (%d images)" % (name, r["n"]))
    print("  images with at least one false positive: %.1f%%  95%% CI [%.1f%%, %.1f%%]"
          % (100 * r["images_with_any_false_positive"], 100 * r["lo"], 100 * r["hi"]))
    print("  mean findings called per image: %.3f" % r["mean_findings_per_image"])
    worst = sorted(r["per_class_false_positive_rate"].items(), key=lambda kv: -kv[1])[:5]
    print("  most often wrong:")
    for c, v in worst:
        print("    %-28s %.1f%%" % (c, 100 * v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--match_size", type=int, default=None,
                    help="long side of the external images, for the resolution control")
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--cpu", action="store_true", help="leave the GPU to a training run")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    th = load_thresholds()
    print("device: %s   thresholds from: %s" % (device, THRESHOLDS))

    model = build_model(device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print("classifier: %s  (epoch %s, val AUC %.4f)"
          % (args.ckpt, state.get("epoch"), state.get("val_auc", float("nan"))))

    ext = pd.read_csv(args.manifest)
    for c in LABEL_COLS:
        if c not in ext.columns:
            ext[c] = 0
    print("external set: %d images from %s" % (len(ext), args.manifest))

    _, _, te = load_frames(".")
    mimic_normal = te[te[NORMAL] == 1].reset_index(drop=True)
    print("MIMIC test normals: %d images" % len(mimic_normal))

    rng = np.random.default_rng(SEED)
    out = {}

    tf = eval_transform(args.img_size)
    out["external"] = rates(flags(predict(model, ext, tf, device), th), rng)
    report("external normals", out["external"])

    out["mimic"] = rates(flags(predict(model, mimic_normal, tf, device), th), rng)
    report("MIMIC normals", out["mimic"])

    if args.match_size:
        tf_small = eval_transform(args.img_size, match_size=args.match_size)
        out["mimic_resolution_matched"] = rates(
            flags(predict(model, mimic_normal, tf_small, device), th), rng)
        report("MIMIC normals shrunk to long side %d" % args.match_size,
               out["mimic_resolution_matched"])

    e = out["external"]["images_with_any_false_positive"]
    m = out["mimic"]["images_with_any_false_positive"]
    print("\nexternal minus MIMIC: %+.1f percentage points" % (100 * (e - m)))
    if args.match_size:
        r = out["mimic_resolution_matched"]["images_with_any_false_positive"]
        print("resolution alone accounts for: %+.1f points" % (100 * (r - m)))
        print("domain beyond resolution:      %+.1f points" % (100 * (e - r)))

    with open("external_eval.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved external_eval.json")


if __name__ == "__main__":
    main()