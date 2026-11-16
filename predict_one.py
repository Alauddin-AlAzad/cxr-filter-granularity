"""
Runs the full pipeline on a single image.

    python predict_one.py path/to/image.jpg

Prints the findings the classifier detected and the report the generator wrote.
"""

import json
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import densenet121

from rgp_data import (Vocab, make_prompt_ids, eval_tf, LABEL_COLS,
                      PAD, BOS, EOS, SEP, MAX_LEN, PROMPT_LEN)

DATA_DIR = "."
CLS_CKPT = "checkpoints/best.pt"
GEN_DIR = "checkpoints_prompt"
BEAM = 3
MAX_FINDINGS = 3

D_MODEL, N_HEAD, N_LAYERS, FF_DIM = 512, 8, 3, 2048
TOTAL_LEN = PROMPT_LEN + MAX_LEN


class PromptedGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        backbone = densenet121(weights=None)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, 13)
        self.features = backbone.features
        self.proj = nn.Linear(1024, D_MODEL)
        self.img_pos = nn.Parameter(torch.zeros(1, 49, D_MODEL))
        self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD)
        self.txt_pos = nn.Parameter(torch.zeros(1, TOTAL_LEN, D_MODEL))
        self.drop = nn.Dropout(0.0)
        layer = nn.TransformerDecoderLayer(D_MODEL, N_HEAD, FF_DIM, 0.0,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, N_LAYERS)
        self.out = nn.Linear(D_MODEL, vocab_size)

    def encode(self, img):
        f = self.features(img)
        f = torch.relu(f).flatten(2).transpose(1, 2)
        return self.proj(f) + self.img_pos

    def decode(self, memory, tgt_in):
        t = tgt_in.size(1)
        x = self.drop(self.emb(tgt_in) + self.txt_pos[:, :t])
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tgt_in.device), 1)
        return self.out(self.decoder(x, memory, tgt_mask=mask))

    @torch.no_grad()
    def generate(self, img, prefix, beam=BEAM, length_penalty=1.0):
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

        for _ in range(MAX_LEN):
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


def main(image_path):
    if not os.path.isfile(image_path):
        print("File not found:", image_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path}  ({img.size[0]}x{img.size[1]})")
    x = eval_tf(img).unsqueeze(0).to(device)

    clf = densenet121(weights=None)
    clf.classifier = nn.Linear(clf.classifier.in_features, len(LABEL_COLS))
    clf.load_state_dict(torch.load(CLS_CKPT, map_location="cpu", weights_only=False)["model"])
    clf = clf.to(device).eval()

    with torch.no_grad():
        probs = torch.sigmoid(clf(x))[0].cpu().numpy()

    thresholds = json.load(open(os.path.join(DATA_DIR, "prompt_thresholds.json")))

    print("\nClassifier output:")
    order = probs.argsort()[::-1]
    for i in order:
        c = LABEL_COLS[i]
        th = thresholds.get(c, 0.5)
        mark = "  <-- prompted" if (th <= 1.0 and probs[i] >= th and c != "No Finding") else ""
        th_txt = "never" if th > 1.0 else f"{th:.2f}"
        print(f"  {c:30s} p={probs[i]:.3f}   threshold={th_txt}{mark}")

    hits = [(LABEL_COLS[i], probs[i]) for i in range(1, len(LABEL_COLS))
            if thresholds.get(LABEL_COLS[i], 0.5) <= 1.0
            and probs[i] >= thresholds.get(LABEL_COLS[i], 0.5)]
    hits.sort(key=lambda t: -t[1])
    findings = [c for c, _ in hits[:MAX_FINDINGS]]
    if not findings:
        findings = ["No Finding"]

    print("\nPrompt given to the generator:", " | ".join(findings))

    vocab = Vocab.load(os.path.join(GEN_DIR, "vocab.json"))
    gen = PromptedGenerator(len(vocab))
    gen.load_state_dict(torch.load(os.path.join(GEN_DIR, "best.pt"),
                                   map_location="cpu", weights_only=False)["model"])
    gen = gen.to(device).eval()

    prefix = torch.tensor([make_prompt_ids(vocab, findings)], device=device)
    out = gen.generate(x, prefix)
    report = " ".join(vocab.decode(out[0].cpu()))

    print("\n" + "=" * 70)
    print("GENERATED REPORT")
    print("=" * 70)
    print(report if report else "(empty)")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_one.py path/to/image.jpg")
    else:
        main(sys.argv[1])