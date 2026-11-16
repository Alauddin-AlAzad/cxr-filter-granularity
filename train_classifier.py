"""
DenseNet121 multi-label classifier on the MIMIC-CXR subset.

Reads the transforms and splits from cxr_data.py, so the 320 px padded input is
whatever that file defines. Writes to checkpoints_320/ and leaves the earlier
224 px checkpoints untouched, so both can be compared and later ensembled.

    python train_classifier.py
    python train_classifier.py --eval

Safe to stop and restart: the last epoch is saved after every epoch and the run
resumes from it.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, densenet121

from cxr_data import (CXRDataset, LABEL_COLS, NORM, PadSquare, load_frames, pos_weight, IMG_SIZE)
from torchvision import transforms

DATA_DIR = "."
CKPT_DIR = "checkpoints_320"
EPOCHS = 6
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2
CLIP = 1.0


def make_transforms(img_size, train):
    steps = [PadSquare(), transforms.Resize((img_size, img_size))]
    if train:
        steps.append(transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.97, 1.03)))
    steps += [transforms.ToTensor(), NORM]
    return transforms.Compose(steps)


def build_model(device):
    m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, len(LABEL_COLS))
    return m.to(device)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        P.append(torch.sigmoid(out.float()).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def auc_table(probs, y):
    rows = []
    for i, c in enumerate(LABEL_COLS):
        col = y[:, i]
        if col.min() == col.max():
            rows.append((c, float("nan"), int(col.sum())))
        else:
            rows.append((c, roc_auc_score(col, probs[:, i]), int(col.sum())))
    vals = [a for _, a, _ in rows if a == a]
    return rows, float(np.mean(vals)) if vals else 0.0


def print_aucs(rows, mean, title):
    print("\n%s" % title)
    print("  %-28s %8s %8s" % ("", "AUC", "positives"))
    for c, a, n in rows:
        print("  %-28s %8s %8d" % (c, "n/a" if a != a else "%.3f" % a, n))
    print("  %-28s %8.4f" % ("MEAN", mean))


def save_atomic(path, obj):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--ckpt_dir", default=None)
    args = ap.parse_args()

    ckpt_dir = args.ckpt_dir or ("checkpoints_%d" % args.img_size)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = load_frames(DATA_DIR)
    train_tf = make_transforms(args.img_size, True)
    eval_tf = make_transforms(args.img_size, False)
    print("train %d  validate %d  test %d   input %dx%d   -> %s"
          % (len(tr), len(va), len(te), args.img_size, args.img_size, ckpt_dir))

    model = build_model(device)

    val_loader = DataLoader(CXRDataset(va, eval_tf), batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(CXRDataset(te, eval_tf), batch_size=args.batch, shuffle=False,
                             num_workers=0, pin_memory=True)

    if args.eval:
        if not os.path.isfile(best_path):
            print("no checkpoint at", best_path)
            return
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
        evaluate(model, val_loader, test_loader, device, ckpt_dir)
        return

    train_loader = DataLoader(CXRDataset(tr, train_tf), batch_size=args.batch, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=NUM_WORKERS > 0)

    pw = pos_weight(tr).to(device)
    print("pos_weight range: %.2f to %.2f" % (pw.min().item(), pw.max().item()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best = 0, -1.0
    if os.path.isfile(last_path):
        try:
            st = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"])
            start_epoch = st["epoch"] + 1
            best = st["best"]
            print("resumed from epoch %d  (best val AUC %.4f)" % (st["epoch"], best))
        except Exception as e:
            print("could not resume (%s), starting fresh" % e)

    for ep in range(start_epoch, args.epochs):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(x.to(device, non_blocking=True)), y.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            seen += 1
            if (i + 1) % 200 == 0:
                done = (i + 1) * args.batch
                rate = done / (time.time() - t0)
                print("  epoch %d  %d/%d  loss %.4f  %.0f img/s  eta %.0f min"
                      % (ep, i + 1, len(train_loader), run / seen, rate,
                         (len(tr) - done) / rate / 60), flush=True)
        sched.step()

        probs, y = predict(model, val_loader, device)
        rows, mean = auc_table(probs, y)
        print("epoch %d  train loss %.4f  val mean AUC %.4f  (%.0f min)"
              % (ep, run / max(seen, 1), mean, (time.time() - t0) / 60))

        save_atomic(last_path, {"model": model.state_dict(), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                                "epoch": ep, "best": max(best, mean)})
        if mean > best:
            best = mean
            save_atomic(best_path, {"model": model.state_dict(), "epoch": ep, "val_auc": mean})
            print("  saved best")

    print("\nloading best checkpoint")
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    evaluate(model, val_loader, test_loader, device, ckpt_dir)


def evaluate(model, val_loader, test_loader, device, ckpt_dir):
    vp, vy = predict(model, val_loader, device)
    vrows, vmean = auc_table(vp, vy)
    print_aucs(vrows, vmean, "validation")

    tp, ty = predict(model, test_loader, device)
    trows, tmean = auc_table(tp, ty)
    print_aucs(trows, tmean, "test")

    out = {"val_mean_auc": round(vmean, 4), "test_mean_auc": round(tmean, 4),
           "val_per_class": {c: None if a != a else round(a, 4) for c, a, _ in vrows},
           "test_per_class": {c: None if a != a else round(a, 4) for c, a, _ in trows}}
    with open(os.path.join(ckpt_dir, "auc.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved %s/auc.json" % ckpt_dir)


if __name__ == "__main__":
    main()