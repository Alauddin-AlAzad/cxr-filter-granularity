"""
Evaluates the MIMIC-trained system on the IN-CXR Indian chest radiographs.

IN-CXR carries only a normal/abnormal label and no reports, so this measures
three things that label supports:

  1. how well the classifier separates normal from abnormal
  2. how often it calls a finding, on normal and on abnormal radiographs,
     against the same figures on MIMIC
  3. how many findings the generator writes into a report, again for both
     kinds of radiograph and both datasets

The third is the one the literature has no number for. Normals alone cannot
tell a lower false-positive rate from a collapse onto the normal template, so
the abnormal half is what separates the two.

    python incxr_eval.py
    python incxr_eval.py --cpu --gen_n 400

Writes incxr_eval.json and incxr_for_radiologist.csv.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from cxr_data import CXRDataset, LABEL_COLS, NORM, PadSquare, load_frames
from train_classifier import build_model
from clinical_eval import extract_findings, LABELS
from gen_data import MAX_FINDINGS

CLS_CKPT = "checkpoints_320/best.pt"
GEN_TAG = "phrase224"
THRESHOLDS = "prompt_thresholds.json"
MANIFEST = "external/in_cxr/manifest.csv"
IMG_SIZE = 320
GEN_IMG_SIZE = 224
BATCH = 16
N_BOOT = 1000
SEED = 0
NORMAL = "No Finding"
DISEASES = [c for c in LABEL_COLS if c != NORMAL]


def eval_transform(img_size):
    return transforms.Compose([PadSquare(), transforms.Resize((img_size, img_size)),
                               transforms.ToTensor(), NORM])


def load_thresholds():
    if not os.path.isfile(THRESHOLDS):
        print("no %s, using 0.5" % THRESHOLDS)
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


def boot_ci(values, fn, rng, n_boot=N_BOOT):
    n = len(values[0])
    point = fn(*values, np.arange(n))
    draws = [fn(*values, rng.integers(0, n, n)) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def flag_matrix(probs, th):
    idx = [LABEL_COLS.index(c) for c in DISEASES]
    cut = np.array([th[c] for c in DISEASES])
    return probs[:, idx] >= cut


def findings_from_probs(probs, th):
    """The same prompt the generator was trained on: the findings the classifier
    calls, strongest first. Built here from this run's probabilities so both
    datasets are prompted the same way."""
    out = []
    for row in probs:
        hits = [(c, row[LABEL_COLS.index(c)]) for c in DISEASES if row[LABEL_COLS.index(c)] >= th[c]]
        hits.sort(key=lambda x: -x[1])
        out.append([c for c, _ in hits[:MAX_FINDINGS]])
    return out


def rate(flags, idx):
    return float(flags[idx].any(axis=1).mean())


def auc_of(score, y, idx):
    s, t = score[idx], y[idx]
    if t.min() == t.max():
        return float("nan")
    return float(roc_auc_score(t, s))


# ---------------------------------------------------------------- generation


@torch.no_grad()
def generate_reports(df, findings, device, img_size, limit, beam, rng):
    """Returns the generated reports and the prompt each one was given, so the
    two can be compared on exactly the same images.

    The images are drawn at random rather than taken from the front of the
    file, which would follow the order the folders were converted in.
    """
    from gen_data import ReportDataset, Vocab, collate, make_transforms
    from train_generator import PromptedGenerator

    ckpt_dir = "checkpoints_gen_" + GEN_TAG
    vocab = Vocab.load(os.path.join(ckpt_dir, "vocab.json"))
    n_tokens = (img_size // 32) ** 2
    model = PromptedGenerator(len(vocab), None, n_tokens=n_tokens).to(device)
    state = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    take = min(limit, len(df))
    pick = np.sort(rng.choice(len(df), size=take, replace=False))
    d = df.iloc[pick].reset_index(drop=True)
    prompts = [findings[i] for i in pick]
    d["report"] = ""
    d["findings"] = prompts
    loader = DataLoader(ReportDataset(d, vocab, make_transforms(img_size, False)),
                        batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)
    hyps = []
    for b in loader:
        out = model.generate(b["img"].to(device), b["prompt"].to(device), beam=beam)
        for row in out:
            hyps.append(" ".join(vocab.decode(row.cpu())))
        if len(hyps) % 160 == 0:
            print("    generated %d/%d" % (len(hyps), take), flush=True)
    return hyps, prompts, pick


def findings_per_report(hyps):
    counts = [sum(1 for v in extract_findings(h).values() if v) for h in hyps]
    return np.array(counts, dtype=float)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gen_n", type=int, default=400, help="reports to generate per group")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--radiologist_n", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    rng = np.random.default_rng(SEED)
    th = load_thresholds()
    out = {}

    ext = pd.read_csv(args.manifest)
    for c in LABEL_COLS:
        if c not in ext.columns:
            ext[c] = 0
    ext_norm = ext[ext["abnormal"] == 0].reset_index(drop=True)
    ext_abn = ext[ext["abnormal"] == 1].reset_index(drop=True)
    print("IN-CXR: %d normal, %d abnormal" % (len(ext_norm), len(ext_abn)))

    _, _, te = load_frames(".")
    mimic_norm = te[te[NORMAL] == 1].reset_index(drop=True)
    dis = te[DISEASES].fillna(0).replace(-1, 0).values.astype(int)
    mimic_abn = te[(te[NORMAL] != 1) & (dis.sum(axis=1) > 0)].reset_index(drop=True)
    print("MIMIC test: %d normal, %d abnormal" % (len(mimic_norm), len(mimic_abn)))

    model = build_model(device)
    state = torch.load(CLS_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print("classifier: %s  (val AUC %.4f)\n" % (CLS_CKPT, state.get("val_auc", float("nan"))))

    tf = eval_transform(IMG_SIZE)
    p_norm = predict(model, ext_norm, tf, device)
    p_abn = predict(model, ext_abn, tf, device)
    p_mimic = predict(model, mimic_norm, tf, device)
    p_mimic_abn = predict(model, mimic_abn, tf, device)

    # 1. normal vs abnormal
    ni = LABEL_COLS.index(NORMAL)
    score = np.concatenate([1 - p_norm[:, ni], 1 - p_abn[:, ni]])
    truth = np.concatenate([np.zeros(len(p_norm)), np.ones(len(p_abn))])
    a, lo, hi = boot_ci((score, truth), auc_of, rng)
    print("normal vs abnormal, 1 - P(No Finding):  AUC %.4f  [%.4f, %.4f]" % (a, lo, hi))
    out["auc_no_finding"] = {"value": round(a, 4), "lo": round(lo, 4), "hi": round(hi, 4)}

    maxdis = np.concatenate([p_norm, p_abn])[:, [LABEL_COLS.index(c) for c in DISEASES]].max(axis=1)
    a2, lo2, hi2 = boot_ci((maxdis, truth), auc_of, rng)
    print("normal vs abnormal, max disease prob:   AUC %.4f  [%.4f, %.4f]" % (a2, lo2, hi2))
    out["auc_max_disease"] = {"value": round(a2, 4), "lo": round(lo2, 4), "hi": round(hi2, 4)}

    # 2. calling a finding on a normal radiograph
    print("\nimages with at least one finding called, at the MIMIC thresholds")
    groups = [("IN-CXR normal", p_norm, ext_norm), ("MIMIC normal", p_mimic, mimic_norm),
              ("IN-CXR abnormal", p_abn, ext_abn), ("MIMIC abnormal", p_mimic_abn, mimic_abn)]
    for name, probs, _ in groups:
        fl = flag_matrix(probs, th)
        r, l, h = boot_ci((fl,), rate, rng)
        print("  %-16s %.1f%%  [%.1f%%, %.1f%%]   mean findings %.2f"
              % (name, 100 * r, 100 * l, 100 * h, fl.sum(axis=1).mean()))
        out["flag_rate_" + name.lower().replace("-", "").replace(" ", "_")] = {
            "value": round(r, 4), "lo": round(l, 4), "hi": round(h, 4),
            "mean_findings": round(float(fl.sum(axis=1).mean()), 3)}

    # 3. findings written into generated reports
    gen_dir = "checkpoints_gen_" + GEN_TAG
    if os.path.isfile(os.path.join(gen_dir, "best.pt")):
        print("\ngenerating reports with %s" % GEN_TAG)
        rng_pick = np.random.default_rng(SEED + 1)
        hyps, prompted, picked = {}, {}, {}
        for name, probs, frame in groups:
            print("  %s" % name, flush=True)
            fnd = findings_from_probs(probs, th)
            hyps[name], prompted[name], picked[name] = generate_reports(
                frame, fnd, device, GEN_IMG_SIZE, args.gen_n, args.beam, rng_pick)

        print("\nfindings written per report")
        for name, _, _ in groups:
            c = findings_per_report(hyps[name])
            m, l, h = boot_ci((c,), lambda v, idx: float(v[idx].mean()), rng)
            any_r = float((c > 0).mean())
            print("  %-16s %.3f per report  [%.3f, %.3f]   reports naming any finding %.1f%%"
                  % (name, m, l, h, 100 * any_r))
            out["gen_findings_" + name.lower().replace("-", "").replace(" ", "_")] = {
                "n": len(hyps[name]), "mean": round(m, 4), "lo": round(l, 4),
                "hi": round(h, 4), "any_finding_rate": round(any_r, 4)}

        # does the report keep what the prompt handed it, on the same images
        print("\nprompt the classifier gave vs what the report kept")
        for name, _, _ in groups:
            pr, hy = prompted[name], hyps[name]
            n = min(len(pr), len(hy))
            has = np.array([len(pr[i]) > 0 for i in range(n)])
            if not has.any():
                print("  %-16s no image was prompted with a finding" % name)
                continue
            said = [extract_findings(hy[i]) for i in range(n)]
            kept = np.array([any(said[i].get(c) for c in pr[i]) for i in range(n)])
            sub = kept[has]
            k, l, h = boot_ci((sub.astype(float),),
                              lambda v, idx: float(v[idx].mean()), rng)
            print("  %-16s prompted %d/%d images;  prompt named in the report "
                  "%.1f%%  [%.1f%%, %.1f%%]"
                  % (name, int(has.sum()), n, 100 * k, 100 * l, 100 * h))
            out["prompt_kept_" + name.lower().replace("-", "").replace(" ", "_")] = {
                "n_prompted": int(has.sum()), "n": n, "value": round(k, 4),
                "lo": round(l, 4), "hi": round(h, 4)}

        # half normal, half abnormal, so the ratings cover both kinds of error
        half = max(args.radiologist_n // 2, 1)
        picks = []
        for name, frame in [("IN-CXR normal", ext_norm), ("IN-CXR abnormal", ext_abn)]:
            h = hyps[name]
            keep = frame.iloc[picked[name][: len(h)]].reset_index(drop=True)
            keep["generated_report"] = h
            keep["truth"] = "abnormal" if "abnormal" in name else "normal"
            picks.append(keep.sample(n=min(half, len(keep)), random_state=SEED))
        sample = pd.concat(picks).sample(frac=1.0, random_state=SEED)
        sample["case"] = range(1, len(sample) + 1)
        # the label stays out of the rating sheet and goes in a separate key
        sample[["case", "image_id", "local_path", "generated_report"]].to_csv(
            "incxr_for_radiologist.csv", index=False)
        sample[["case", "image_id", "truth"]].to_csv("incxr_radiologist_key.csv", index=False)
        print("\nsaved incxr_for_radiologist.csv (%d rows) and incxr_radiologist_key.csv"
              % len(sample))
    else:
        print("\nno generator checkpoint at %s, skipping the report part" % gen_dir)

    with open("incxr_eval.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved incxr_eval.json")


if __name__ == "__main__":
    main()