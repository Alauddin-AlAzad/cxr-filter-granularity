"""
Prompted report generator: DenseNet-121 encoder + 3-layer Transformer decoder.

The encoder is initialised from the trained classifier. Its feature grid becomes
visual tokens the decoder cross-attends to. A short prompt naming the findings is
prepended to the decoder input and masked out of the loss.

One script covers every row of the ablation:

    python train_generator.py --prompts calibrated      main system
    python train_generator.py --prompts uncalibrated    calibration ablation
    python train_generator.py --prompts oracle          upper bound
    python train_generator.py --prompts none            no prompting
    python train_generator.py --prompts calibrated --reports unfiltered
    python train_generator.py --prompts uncalibrated --reports phrase --seed 2

Each combination writes to its own checkpoint folder, so nothing overwrites
anything. Add --eval to score an existing checkpoint without training.
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import densenet121

from cxr_data import IMG_SIZE, LABEL_COLS
from gen_data import (MAX_LEN, PAD, PROMPT_LEN, BOS, EOS, ReportDataset, Vocab,
                      collate, load_splits, make_transforms)

DATA_DIR = "."
CLS_CKPT = "checkpoints_320/best.pt"

EPOCHS = 8
BATCH_SIZE = 16
LR = 1e-4
ENCODER_LR = 1e-5
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2
CLIP = 1.0
LABEL_SMOOTH = 0.1

D_MODEL, N_HEAD, N_LAYERS, FF_DIM, DROPOUT = 512, 8, 3, 2048, 0.1
MAX_TOKENS = 144  # enough for inputs up to 384 px
TOTAL_LEN = PROMPT_LEN + MAX_LEN

BEAM = 3
VAL_GEN_N = 300


class PromptedGenerator(nn.Module):
    def __init__(self, vocab_size, cls_ckpt=None, dropout=DROPOUT, n_tokens=MAX_TOKENS):
        super().__init__()
        backbone = densenet121(weights=None)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, len(LABEL_COLS))
        if cls_ckpt and os.path.isfile(cls_ckpt):
            backbone.load_state_dict(torch.load(cls_ckpt, map_location="cpu",
                                                weights_only=False)["model"])
            print("encoder initialised from", cls_ckpt)
        self.features = backbone.features
        self.proj = nn.Linear(1024, D_MODEL)
        self.img_pos = nn.Parameter(torch.zeros(1, n_tokens, D_MODEL))
        self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD)
        self.txt_pos = nn.Parameter(torch.zeros(1, TOTAL_LEN, D_MODEL))
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerDecoderLayer(D_MODEL, N_HEAD, FF_DIM, dropout,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, N_LAYERS)
        self.out = nn.Linear(D_MODEL, vocab_size)
        nn.init.trunc_normal_(self.img_pos, std=0.02)
        nn.init.trunc_normal_(self.txt_pos, std=0.02)

    def encode(self, img):
        f = self.features(img)
        f = torch.relu(f).flatten(2).transpose(1, 2)
        return self.proj(f) + self.img_pos[:, : f.size(1)]

    def decode(self, memory, tgt_in):
        t = tgt_in.size(1)
        x = self.drop(self.emb(tgt_in) + self.txt_pos[:, :t])
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tgt_in.device), 1)
        return self.out(self.decoder(x, memory, tgt_mask=mask))

    def forward(self, img, tgt_in):
        return self.decode(self.encode(img), tgt_in)

    @torch.no_grad()
    def generate(self, img, prefix, beam=BEAM, length_penalty=1.0, max_len=MAX_LEN):
        dev = img.device
        memory = self.encode(img)
        b = img.size(0)
        n, d = memory.size(1), memory.size(2)
        V = self.out.out_features
        memory = memory.unsqueeze(1).expand(b, beam, n, d).reshape(b * beam, n, d)

        P = prefix.size(1)
        ids = prefix.unsqueeze(1).expand(b, beam, P).reshape(b * beam, P).clone()
        scores = torch.full((b, beam), -1e9, device=dev)
        scores[:, 0] = 0.0
        scores = scores.view(-1)
        fin = torch.zeros(b * beam, dtype=torch.bool, device=dev)
        ar = torch.arange(b, device=dev).unsqueeze(1)

        for _ in range(max_len):
            lp = torch.log_softmax(self.decode(memory, ids)[:, -1].float(), -1)
            lp[fin] = -1e9
            lp[fin, PAD] = 0.0
            cand = (scores.unsqueeze(1) + lp).view(b, beam * V)
            top, idx = cand.topk(beam, dim=-1)
            bi = torch.div(idx, V, rounding_mode="floor")
            tok = idx % V
            g = (ar * beam + bi).view(-1)
            ids = torch.cat([ids[g], tok.view(-1, 1)], dim=1)
            scores = top.reshape(-1)
            fin = fin[g] | (tok.view(-1) == EOS)
            if fin.all():
                break

        gen = ids[:, P:]
        lens = (gen != PAD).sum(1).clamp(min=1).float()
        best = (scores / lens.pow(length_penalty)).view(b, beam).argmax(-1)
        return gen[torch.arange(b, device=dev) * beam + best]


# ---------------------------------------------------------------- metrics


def ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    hl = sum(len(h) for h in hyps)
    rl = sum(len(r) for r in refs)
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    out, logs = [], []
    for n in range(1, max_n + 1):
        num = den = 0
        for h, r in zip(hyps, refs):
            hn, rn = ngrams(h, n), ngrams(r, n)
            num += sum(min(c, rn[g]) for g, c in hn.items())
            den += sum(hn.values())
        p = num / den if den else 0.0
        logs.append(math.log(p) if p > 0 else -1e9)
        out.append(bp * math.exp(sum(logs[:n]) / n) if p > 0 else 0.0)
    return out


def rouge_l(hyps, refs):
    scores = []
    for h, r in zip(hyps, refs):
        if not h or not r:
            scores.append(0.0)
            continue
        m, n = len(h), len(r)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                dp[i + 1][j + 1] = dp[i][j] + 1 if h[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
        l = dp[m][n]
        p, rc = l / m, l / n
        scores.append(0.0 if p + rc == 0 else 2 * p * rc / (p + rc))
    return float(np.mean(scores)) if scores else 0.0


def meteor(hyps, refs):
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        return float(np.mean([meteor_score([r.lower().split()], h.lower().split())
                             for h, r in zip(hyps, refs)]))
    except Exception as e:
        print("  METEOR skipped (%s)" % e)
        return None


def cider(hyps, refs, n=4, sigma=6.0):
    """CIDEr-D with a single reference per image."""
    def counts(toks):
        c = []
        for k in range(1, n + 1):
            c.append(ngrams(toks, k))
        return c

    hc = [counts(h) for h in hyps]
    rc = [counts(r) for r in refs]

    df = [Counter() for _ in range(n)]
    for c in rc:
        for k in range(n):
            for g in c[k]:
                df[k][g] += 1
    N = len(refs)

    def vec(c, k):
        v, norm = {}, 0.0
        for g, f in c[k].items():
            idf = math.log(max(N, 1)) - math.log(max(df[k].get(g, 0), 1))
            v[g] = f * idf
            norm += v[g] ** 2
        return v, math.sqrt(norm)

    scores = []
    for ci, cj, h, r in zip(hc, rc, hyps, refs):
        total = 0.0
        for k in range(n):
            vh, nh = vec(ci, k)
            vr, nr = vec(cj, k)
            num = sum(min(vh[g], vr.get(g, 0.0)) * vr.get(g, 0.0) for g in vh)
            denom = nh * nr
            s = num / denom if denom > 0 else 0.0
            delta = len(h) - len(r)
            s *= math.exp(-delta ** 2 / (2 * sigma ** 2))
            total += s
        scores.append(10.0 * total / n)
    return float(np.mean(scores)) if scores else 0.0


def text_metrics(hyps, refs, with_meteor=True):
    h = [x.lower().split() for x in hyps]
    r = [x.lower().split() for x in refs]
    b = corpus_bleu(h, r)
    m = {"BLEU-1": round(b[0], 4), "BLEU-2": round(b[1], 4),
         "BLEU-3": round(b[2], 4), "BLEU-4": round(b[3], 4),
         "ROUGE-L": round(rouge_l(h, r), 4),
         "CIDEr": round(cider(h, r), 4),
         "mean_len": round(float(np.mean([len(x) for x in h])), 1),
         "empty": round(float(np.mean([len(x) == 0 for x in h])), 4)}
    if with_meteor:
        mt = meteor(hyps, refs)
        if mt is not None:
            m["METEOR"] = round(mt, 4)
    return m


# ---------------------------------------------------------------- loops


def step_loss(model, batch, criterion, device):
    prompt = batch["prompt"].to(device)
    report = batch["report"].to(device)
    full = torch.cat([prompt, report], dim=1)
    tgt_in = full[:, :-1]
    tgt_out = full[:, 1:].clone()
    tgt_out[:, : prompt.size(1) - 1] = PAD
    logits = model(batch["img"].to(device, non_blocking=True), tgt_in)
    return criterion(logits.reshape(-1, logits.size(-1)).float(), tgt_out.reshape(-1))


@torch.no_grad()
def val_loss(model, loader, criterion, device):
    model.eval()
    tot = n = 0
    for b in loader:
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            tot += step_loss(model, b, criterion, device).item()
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def run_generation(model, df, vocab, device, limit=None, beam=BEAM, batch=8, length_penalty=1.0,
                   img_size=IMG_SIZE):
    d = df if limit is None else df.iloc[:limit]
    loader = DataLoader(ReportDataset(d, vocab, make_transforms(img_size, False)),
                        batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate)
    model.eval()
    hyps, refs = [], []
    t0 = time.time()
    for bi, b in enumerate(loader, 1):
        out = model.generate(b["img"].to(device), b["prompt"].to(device), beam=beam,
                             length_penalty=length_penalty)
        for row in out:
            hyps.append(" ".join(vocab.decode(row.cpu())))
        refs.extend(b["refs"])
        if bi % 20 == 0:
            print("    generated %d/%d  (%.0fs)" % (len(hyps), len(d), time.time() - t0), flush=True)
    return hyps, refs


def save_atomic(path, obj):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="calibrated",
                    choices=["calibrated", "uncalibrated", "oracle", "none"])
    ap.add_argument("--reports", choices=["filtered", "unfiltered", "phrase"], default="filtered")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--beam", type=int, default=BEAM)
    ap.add_argument("--lp", type=float, default=1.0, help="length penalty at generation time")
    ap.add_argument("--label_smooth", type=float, default=LABEL_SMOOTH)
    ap.add_argument("--augment", default="affine", choices=["affine", "none"])
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--tag", default=None, help="override the checkpoint folder suffix")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the initialisation and shuffling order, for replication")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print("seed: %d" % args.seed)

    tag = args.tag or "%s_%s" % (args.prompts, args.reports)
    ckpt_dir = "checkpoints_gen_" + tag
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")
    vocab_path = os.path.join(ckpt_dir, "vocab.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = load_splits(DATA_DIR, args.prompts, args.reports)
    print("run: %s" % tag)
    print("train %d  validate %d  test %d" % (len(tr), len(va), len(te)))
    empty = float(np.mean([len(f) == 0 for f in tr["findings"]]))
    print("prompt findings per image: %.2f   empty prompts: %.1f%%"
          % (float(np.mean([len(f) for f in tr["findings"]])), 100 * empty))

    if os.path.isfile(vocab_path):
        vocab = Vocab.load(vocab_path)
    else:
        vocab = Vocab.build(tr["report"].tolist())
        vocab.save(vocab_path)
    print("vocabulary: %d words" % len(vocab))

    n_tokens = (args.img_size // 32) ** 2
    model = PromptedGenerator(len(vocab), CLS_CKPT, n_tokens=n_tokens).to(device)
    print("input %dpx -> %d visual tokens" % (args.img_size, n_tokens))
    print("parameters: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=args.label_smooth)
    print("label smoothing %.2f   augmentation %s" % (args.label_smooth, args.augment))

    if args.eval:
        if not os.path.isfile(best_path):
            print("no checkpoint at", best_path)
            return
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
        evaluate(model, te, vocab, device, ckpt_dir, args.beam, args.lp, args.img_size)
        return

    train_loader = DataLoader(ReportDataset(tr, vocab, make_transforms(args.img_size, True, args.augment == "affine")),
                              batch_size=args.batch, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True, collate_fn=collate,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(ReportDataset(va, vocab, make_transforms(args.img_size, False)),
                            batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)

    enc_ids = {id(p) for p in model.features.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": LR},
        {"params": list(model.features.parameters()), "lr": ENCODER_LR},
    ], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best = 0, 1e9
    if os.path.isfile(last_path):
        try:
            st = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"])
            start_epoch = st["epoch"] + 1
            best = st["best"]
            print("resumed from epoch %d  (best val loss %.4f)" % (st["epoch"], best))
        except Exception as e:
            print("could not resume (%s), starting fresh" % e)

    for ep in range(start_epoch, args.epochs):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for i, b in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
                loss = step_loss(model, b, criterion, device)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            seen += 1
            if (i + 1) % 400 == 0:
                done = (i + 1) * args.batch
                rate = done / (time.time() - t0)
                print("  epoch %d  %d/%d  loss %.4f  %.0f img/s  eta %.0f min"
                      % (ep, i + 1, len(train_loader), run / seen, rate,
                         (len(tr) - done) / rate / 60), flush=True)
        sched.step()

        vl = val_loss(model, val_loader, criterion, device)
        print("epoch %d  train %.4f  val %.4f  (%.0f min)"
              % (ep, run / max(seen, 1), vl, (time.time() - t0) / 60))

        save_atomic(last_path, {"model": model.state_dict(), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                                "epoch": ep, "best": min(best, vl)})
        if vl < best:
            best = vl
            save_atomic(best_path, {"model": model.state_dict(), "epoch": ep, "val_loss": vl})
            print("  saved best")

        if VAL_GEN_N:
            hyps, refs = run_generation(model, va, vocab, device, limit=VAL_GEN_N, beam=1,
                                        img_size=args.img_size)
            print("  greedy on %d val images: %s" % (len(hyps), text_metrics(hyps, refs, False)))

    print("\nloading best checkpoint")
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    evaluate(model, te, vocab, device, ckpt_dir, args.beam, args.lp, args.img_size)


def evaluate(model, te, vocab, device, ckpt_dir, beam, lp=1.0, img_size=IMG_SIZE):
    hyps, refs = run_generation(model, te, vocab, device, beam=beam, length_penalty=lp, img_size=img_size)
    m = text_metrics(hyps, refs)
    print("\ntest text metrics:")
    for k, v in m.items():
        print("  %-10s %s" % (k, v))

    with open(os.path.join(ckpt_dir, "test_generated.json"), "w") as f:
        json.dump([{"hyp": h, "ref": r} for h, r in zip(hyps, refs)], f, indent=1)

    try:
        from clinical_eval import clinical_efficacy, LABELS as CE_LABELS
        true = te[CE_LABELS].fillna(0).replace(-1, 0).values.astype(int)
        per, summary = clinical_efficacy(hyps, true)
        print("\nclinical efficacy (truth = CheXpert labels):")
        print(per.to_string(index=False))
        print("\n ", summary)
        m["clinical"] = {k: float(v) for k, v in summary.items()}
    except Exception as e:
        print("\nclinical efficacy skipped (%s)" % e)

    with open(os.path.join(ckpt_dir, "test_metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    print("\nsaved %s/test_metrics.json" % ckpt_dir)


if __name__ == "__main__":
    main()