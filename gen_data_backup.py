"""
Data module for the prompted report generator.

The vocabulary is built from the training reports only. The prompt is a short
block of finding names followed by a separator, prepended to the report; the
loss is masked over it during training, so the model learns to write a report
consistent with the prompt rather than to reproduce the prompt.

Must stay an importable file, not notebook code, so DataLoader workers can
pickle the Dataset on Windows.
"""

import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms

from cxr_data import IMG_SIZE, LABEL_COLS, NORM, PadSquare, eval_tf, train_tf

MAX_LEN = 100
PROMPT_LEN = 16
MIN_FREQ = 3
MAX_FINDINGS = 3

PAD, BOS, EOS, UNK, SEP = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"]

_TOKEN = re.compile(r"[a-z]+|[0-9]+|[.,;:]")


def tokenize(text):
    return _TOKEN.findall(str(text).lower())


class Vocab:
    def __init__(self, stoi):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self):
        return len(self.stoi)

    @classmethod
    def build(cls, texts, min_freq=MIN_FREQ):
        c = Counter()
        for t in texts:
            c.update(tokenize(t))
        stoi = {s: i for i, s in enumerate(SPECIALS)}
        for w, n in c.most_common():
            if n >= min_freq:
                stoi[w] = len(stoi)
        return cls(stoi)

    def encode(self, text, max_len=MAX_LEN):
        ids = [BOS] + [self.stoi.get(w, UNK) for w in tokenize(text)][: max_len - 2] + [EOS]
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i in (PAD, BOS, SEP):
                continue
            if i == EOS:
                break
            out.append(self.itos.get(i, "<unk>"))
        return out

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stoi, f)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))


def make_prompt_ids(vocab, findings, prompt_len=PROMPT_LEN):
    ids = [BOS]
    for f in findings:
        ids += [vocab.stoi.get(w, UNK) for w in tokenize(f)]
    ids = ids[: prompt_len - 1]
    ids.append(SEP)
    return ids + [PAD] * (prompt_len - len(ids))


# ---------------------------------------------------------------- frames


def find_text_column(df):
    for c in ["report", "text", "clean_report"]:
        if c in df.columns:
            return c
    raise RuntimeError("no report column in image_report_pairs.csv")


def load_pairs(data_dir=".", reports="filtered"):
    name = "image_report_pairs.csv" if reports == "filtered" else "image_report_pairs_raw.csv"
    path = os.path.join(data_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    labels = pd.read_csv(os.path.join(data_dir, "available_labels.csv"))
    pairs = pd.read_csv(path)
    tcol = find_text_column(pairs)
    pairs = pairs[["dicom_id", tcol]].rename(columns={tcol: "report"})
    df = labels.merge(pairs, on="dicom_id", how="inner")
    df = df[df["report"].notna()]
    df = df[df["report"].astype(str).str.split().str.len() >= 4].reset_index(drop=True)
    return df


def attach_prompts(df, data_dir=".", prompts="calibrated", max_findings=MAX_FINDINGS):
    """prompts: calibrated | uncalibrated | oracle | none"""
    diseases = LABEL_COLS[1:]
    df = df.copy()

    if prompts == "none":
        df["findings"] = [[] for _ in range(len(df))]
        return df

    if prompts == "oracle":
        out = []
        for _, r in df.iterrows():
            hits = [c for c in diseases if r.get(c, 0) == 1]
            out.append(hits[:max_findings])
        df["findings"] = out
        return df

    if prompts == "calibrated":
        pfile, tfile = "prompt_probs_calibrated.csv", "prompt_thresholds_calibrated.json"
    else:
        pfile, tfile = "prompt_probs.csv", "prompt_thresholds.json"

    probs = pd.read_csv(os.path.join(data_dir, pfile))
    th = json.load(open(os.path.join(data_dir, tfile)))
    df = df.merge(probs, on="dicom_id", how="left")

    out = []
    for _, r in df.iterrows():
        hits = []
        for c in diseases:
            t = th.get(c, 0.5)
            col = "p_" + c
            if isinstance(t, (int, float)) and t <= 1.0 and col in r and not pd.isna(r[col]) and r[col] >= t:
                hits.append((c, r[col]))
        hits.sort(key=lambda x: -x[1])
        out.append([c for c, _ in hits[:max_findings]])
    df["findings"] = out
    return df


def load_splits(data_dir=".", prompts="calibrated", reports="filtered"):
    df = load_pairs(data_dir, reports)
    df = attach_prompts(df, data_dir, prompts)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "validate"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    return tr, va, te


# ---------------------------------------------------------------- dataset


class ReportDataset(Dataset):
    def __init__(self, df, vocab, tf):
        self.paths = df["local_path"].tolist()
        self.reports = df["report"].astype(str).tolist()
        self.findings = df["findings"].tolist()
        self.vocab = vocab
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        for k in range(10):
            j = (i + k) % len(self.paths)
            try:
                img = Image.open(self.paths[j]).convert("RGB")
            except Exception:
                continue
            prompt = torch.tensor(make_prompt_ids(self.vocab, self.findings[j]))
            rep = self.vocab.encode(self.reports[j])
            rep = rep + [PAD] * (MAX_LEN - len(rep))
            return {"img": self.tf(img),
                    "prompt": prompt,
                    "report": torch.tensor(rep[:MAX_LEN]),
                    "ref": self.reports[j]}
        raise RuntimeError("too many unreadable images near index %d" % i)


def collate(batch):
    return {"img": torch.stack([b["img"] for b in batch]),
            "prompt": torch.stack([b["prompt"] for b in batch]),
            "report": torch.stack([b["report"] for b in batch]),
            "refs": [b["ref"] for b in batch]}


def make_transforms(img_size=IMG_SIZE, train=False, augment=True):
    steps = [PadSquare(), transforms.Resize((img_size, img_size))]
    if train and augment:
        steps.append(transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.97, 1.03)))
    steps += [transforms.ToTensor(), NORM]
    return transforms.Compose(steps)


def transforms_for(train):
    return train_tf if train else eval_tf