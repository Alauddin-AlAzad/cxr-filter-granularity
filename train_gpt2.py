"""
DenseNet121 encoder + pretrained DistilGPT2 decoder, with diagnosis-aware
prompting. The encoder is initialised from the trained classifier and the
decoder from DistilGPT2, so only the cross-attention layers and the visual
projection start from random values.

    python train_gpt2.py              train, then evaluate the best checkpoint
    python train_gpt2.py --eval       evaluate only
    python train_gpt2.py --oracle     use true labels as the prompt

Added for the second-architecture ablation:
    --pairs FILE   report text to train on (sentence-level / phrase-level file)
    --tag NAME     checkpoint folder = checkpoints_gpt2_<NAME>
    --seed N       random seed (default 1)
    python train_gpt2.py --pairs <file> --tag sent --seed 1

Checkpoints go to checkpoints_gpt2/ (or checkpoints_gpt2_oracle/), so the
earlier runs stay untouched.
"""

import argparse
import json
import math
import os
import time
from collections import Counter

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import densenet121
from transformers import GPT2Config, GPT2LMHeadModel

from gpt2_data import (CXRReportDataset, collate_gen, collate_train, eval_tf,
                       get_tokenizer, load_splits, train_tf, LABEL_COLS)

DATA_DIR = "."
CLS_CKPT = "checkpoints/best.pt"

EPOCHS = 6
BATCH_SIZE = 8
ACCUM = 2
LR = 5e-5
ENCODER_LR = 1e-5
WARMUP = 500
WEIGHT_DECAY = 0.01
NUM_WORKERS = 2
MAX_NEW = 100
MIN_NEW = 30
LENGTH_PENALTY = 1.2
BEAM = 3
GEN_BATCH = 8
VAL_GEN_N = 300
CLIP = 1.0


class ImgReportGPT2(nn.Module):
    def __init__(self, cls_ckpt=None):
        super().__init__()
        cfg = GPT2Config.from_pretrained("distilgpt2", add_cross_attention=True)
        self.gpt = GPT2LMHeadModel.from_pretrained("distilgpt2", config=cfg)
        d = cfg.n_embd

        backbone = densenet121(weights=None)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, len(LABEL_COLS))
        if cls_ckpt and os.path.isfile(cls_ckpt):
            sd = torch.load(cls_ckpt, map_location="cpu", weights_only=False)["model"]
            backbone.load_state_dict(sd)
            print("encoder initialised from", cls_ckpt)
        self.features = backbone.features

        self.proj = nn.Linear(1024, d)
        self.img_pos = nn.Parameter(torch.zeros(1, 49, d))
        self.ln = nn.LayerNorm(d)

    def encode(self, img):
        f = self.features(img)
        f = torch.relu(f).flatten(2).transpose(1, 2)
        return self.ln(self.proj(f) + self.img_pos)

    def forward(self, img, input_ids, attention_mask, labels=None):
        mem = self.encode(img)
        return self.gpt(input_ids=input_ids, attention_mask=attention_mask,
                        encoder_hidden_states=mem, labels=labels)

    @torch.no_grad()
    def generate(self, img, input_ids, attention_mask, tok, beam=BEAM, max_new=MAX_NEW,
                 min_new=MIN_NEW, length_penalty=LENGTH_PENALTY):
        mem = self.encode(img).float()
        out = self.gpt.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=mem,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            num_beams=beam,
            no_repeat_ngram_size=0,
            length_penalty=length_penalty,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        return out[:, input_ids.size(1):]


# ---------------------------------------------------------------- metrics


def ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    """Standard corpus BLEU with brevity penalty, one reference per hypothesis."""
    out = []
    hl = sum(len(h) for h in hyps)
    rl = sum(len(r) for r in refs)
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    logs = []
    for n in range(1, max_n + 1):
        num = den = 0
        for h, r in zip(hyps, refs):
            hn, rn = ngrams(h, n), ngrams(r, n)
            num += sum(min(c, rn[g]) for g, c in hn.items())
            den += max(sum(hn.values()), 0)
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


# ---------------------------------------------------------------- loops


def run_generation(model, df, tok, device, limit=None, beam=BEAM,
                   min_new=MIN_NEW, length_penalty=LENGTH_PENALTY):
    d = df if limit is None else df.iloc[:limit]
    ds = CXRReportDataset(d, eval_tf, train=False)
    loader = DataLoader(ds, batch_size=GEN_BATCH, shuffle=False,
                        num_workers=0, collate_fn=collate_gen)
    model.eval()
    hyps, refs = [], []
    t0 = time.time()
    for bi, b in enumerate(loader, 1):
        img = b["img"].to(device)
        ids = b["input_ids"].to(device)
        att = b["attention_mask"].to(device)
        gen = model.generate(img, ids, att, tok, beam=beam,
                             min_new=min_new, length_penalty=length_penalty)
        for g in gen:
            txt = tok.decode(g, skip_special_tokens=True).strip()
            hyps.append(txt)
        refs.extend(b["refs"])
        if bi % 10 == 0:
            print("    generated %d/%d  (%.0fs)" % (len(hyps), len(ds), time.time() - t0))
    return hyps, refs


def text_metrics(hyps, refs):
    h = [x.lower().split() for x in hyps]
    r = [x.lower().split() for x in refs]
    b = corpus_bleu(h, r)
    return {
        "BLEU-1": round(b[0], 4),
        "BLEU-2": round(b[1], 4),
        "BLEU-3": round(b[2], 4),
        "BLEU-4": round(b[3], 4),
        "ROUGE-L": round(rouge_l(h, r), 4),
        "mean_len": round(float(np.mean([len(x) for x in h])), 1),
        "empty": round(float(np.mean([len(x) == 0 for x in h])), 4),
    }


@torch.no_grad()
def val_loss(model, loader, device):
    model.eval()
    tot = n = 0
    for b in loader:
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            out = model(b["img"].to(device), b["input_ids"].to(device),
                        b["attention_mask"].to(device), labels=b["labels"].to(device))
        tot += out.loss.item()
        n += 1
    return tot / max(n, 1)


def save_atomic(path, obj):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tune", action="store_true", help="sweep decoding settings on the validation split")
    ap.add_argument("--beam", type=int, default=BEAM)
    ap.add_argument("--min_new", type=int, default=MIN_NEW)
    ap.add_argument("--lp", type=float, default=LENGTH_PENALTY)
    ap.add_argument("--pairs", default="image_report_pairs.csv",
                    help="CSV with the report text to train and test on")
    ap.add_argument("--tag", default=None,
                    help="checkpoint folder becomes checkpoints_gpt2_<tag>")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.tag:
        ckpt_dir = "checkpoints_gpt2_" + args.tag
    else:
        ckpt_dir = "checkpoints_gpt2_oracle" if args.oracle else "checkpoints_gpt2"
    print("checkpoint folder:", ckpt_dir, "| seed", args.seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer()

    if not os.path.isfile(os.path.join(DATA_DIR, args.pairs)):
        print("report file not found:", args.pairs)
        return
    tr, va, te = load_splits(DATA_DIR, use_oracle=args.oracle, pairs_file=args.pairs)
    print("train %d  val %d  test %d   oracle=%s" % (len(tr), len(va), len(te), args.oracle))
    print("empty prompts in train: %.1f%%" % (100 * (tr["prompt"] == "none").mean()))

    model = ImgReportGPT2(CLS_CKPT).to(device)
    print("parameters: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    if args.eval or args.tune:
        if not os.path.isfile(best_path):
            print("no checkpoint at", best_path)
            return
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
        if args.tune:
            tune_decoding(model, va, tok, device)
        else:
            evaluate(model, te, tok, device, ckpt_dir,
                     beam=args.beam, min_new=args.min_new, length_penalty=args.lp)
        return

    train_loader = DataLoader(CXRReportDataset(tr, train_tf), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_train, drop_last=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(CXRReportDataset(va, eval_tf, train=False), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, collate_fn=collate_train)

    enc_params = list(model.features.parameters())
    enc_ids = {id(p) for p in enc_params}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": rest, "lr": LR},
        {"params": enc_params, "lr": ENCODER_LR},
    ], weight_decay=WEIGHT_DECAY)

    steps = args.epochs * (len(train_loader) // ACCUM)

    def lr_at(s):
        if s < WARMUP:
            return s / max(WARMUP, 1)
        q = (s - WARMUP) / max(steps - WARMUP, 1)
        return 0.5 * (1 + math.cos(math.pi * min(q, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
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
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(train_loader):
            with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
                out = model(b["img"].to(device, non_blocking=True),
                            b["input_ids"].to(device), b["attention_mask"].to(device),
                            labels=b["labels"].to(device))
                loss = out.loss / ACCUM
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run += out.loss.item()
            seen += 1
            if (i + 1) % 200 == 0:
                print("  epoch %d  %d/%d  loss %.4f  ppl %.1f  %.0fs"
                      % (ep, i + 1, len(train_loader), run / seen,
                         math.exp(min(run / seen, 20)), time.time() - t0))

        vl = val_loss(model, val_loader, device)
        print("epoch %d  train %.4f  val %.4f  ppl %.1f  (%.0f min)"
              % (ep, run / max(seen, 1), vl, math.exp(min(vl, 20)), (time.time() - t0) / 60))

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": ep, "best": min(best, vl)}
        save_atomic(last_path, state)
        if vl < best:
            best = vl
            save_atomic(best_path, {"model": model.state_dict(), "epoch": ep, "val_loss": vl})
            print("  saved best")

        if VAL_GEN_N:
            hyps, refs = run_generation(model, va, tok, device, limit=VAL_GEN_N, beam=1, min_new=0)
            print("  greedy on %d val images: %s" % (len(hyps), text_metrics(hyps, refs)))

    print("\nloading best checkpoint for the final evaluation")
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    evaluate(model, te, tok, device, ckpt_dir)


def tune_decoding(model, va, tok, device, n=300):
    """Finds the decoding settings that score best, without retraining anything."""
    rows = []
    for beam in [1, 3]:
        for min_new in [0, 20, 30, 40, 50]:
            for lp in ([1.0] if beam == 1 else [1.0, 1.5, 2.0]):
                hyps, refs = run_generation(model, va, tok, device, limit=n, beam=beam,
                                            min_new=min_new, length_penalty=lp)
                m = text_metrics(hyps, refs)
                m.update({"beam": beam, "min_new": min_new, "lp": lp})
                rows.append(m)
                print("beam %d  min_new %2d  lp %.1f  ->  BLEU-1 %.4f  BLEU-4 %.4f  ROUGE-L %.4f  len %.1f"
                      % (beam, min_new, lp, m["BLEU-1"], m["BLEU-4"], m["ROUGE-L"], m["mean_len"]))
    best = max(rows, key=lambda r: r["BLEU-4"])
    print("\nbest by BLEU-4: beam %d  min_new %d  lp %.1f" % (best["beam"], best["min_new"], best["lp"]))
    print("run:  python train_gpt2.py --eval --beam %d --min_new %d --lp %.1f"
          % (best["beam"], best["min_new"], best["lp"]))


def evaluate(model, te, tok, device, ckpt_dir, beam=BEAM, min_new=MIN_NEW, length_penalty=LENGTH_PENALTY):
    hyps, refs = run_generation(model, te, tok, device, beam=beam,
                                min_new=min_new, length_penalty=length_penalty)
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
        m["clinical"] = summary
    except Exception as e:
        print("\nclinical efficacy skipped (%s)" % e)

    with open(os.path.join(ckpt_dir, "test_metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    print("\nsaved %s/test_metrics.json and test_generated.json" % ckpt_dir)


if __name__ == "__main__":
    main()