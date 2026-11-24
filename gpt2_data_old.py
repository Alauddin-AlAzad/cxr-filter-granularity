"""
Data module for the DistilGPT2 report generator.

Same images, same splits and same prompts as rgp_data.py, but the text is
tokenised with the GPT-2 tokenizer instead of a vocabulary built from this
dataset. The prompt is written as plain words and the loss is masked over it.

Must stay an importable file, not notebook code, so DataLoader workers can
pickle the Dataset on Windows.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import GPT2TokenizerFast

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMG_SIZE = 224
MAX_REPORT_TOKENS = 110
MAX_FINDINGS = 3

LABEL_COLS = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
]

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    NORM,
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    NORM,
])

_TOK = None


def get_tokenizer():
    global _TOK
    if _TOK is None:
        t = GPT2TokenizerFast.from_pretrained("distilgpt2")
        t.pad_token = t.eos_token
        _TOK = t
    return _TOK


def find_text_column(df):
    for c in ["report", "text", "findings_impression", "clean_report"]:
        if c in df.columns:
            return c
    cands = [c for c in df.columns if df[c].dtype == object and c not in ("dicom_id", "study_id", "local_path", "split")]
    if not cands:
        raise RuntimeError("no text column in image_report_pairs.csv: %s" % list(df.columns))
    return cands[-1]


def load_pairs(data_dir="."):
    labels = pd.read_csv(os.path.join(data_dir, "available_labels.csv"))
    pairs = pd.read_csv(os.path.join(data_dir, "image_report_pairs.csv"))
    tcol = find_text_column(pairs)
    pairs = pairs[["dicom_id", tcol]].rename(columns={tcol: "report"})
    df = labels.merge(pairs, on="dicom_id", how="inner")
    df = df[df["report"].notna()]
    df = df[df["report"].str.split().str.len() >= 4].reset_index(drop=True)
    return df


def attach_prompts(df, data_dir=".", use_oracle=False, max_findings=MAX_FINDINGS):
    """Adds a 'prompt' column holding the finding names, comma separated."""
    diseases = LABEL_COLS[1:]

    if use_oracle:
        texts = []
        for _, r in df.iterrows():
            hits = [c for c in diseases if r.get(c, 0) == 1]
            texts.append(", ".join(h.lower() for h in hits[:max_findings]) if hits else "none")
        df = df.copy()
        df["prompt"] = texts
        return df

    probs = pd.read_csv(os.path.join(data_dir, "prompt_probs.csv"))
    th = json.load(open(os.path.join(data_dir, "prompt_thresholds.json")))
    df = df.merge(probs, on="dicom_id", how="left")

    texts = []
    for _, r in df.iterrows():
        hits = []
        for c in diseases:
            t = th.get(c, 0.5)
            col = "p_" + c
            if t <= 1.0 and col in r and not pd.isna(r[col]) and r[col] >= t:
                hits.append((c, r[col]))
        hits.sort(key=lambda x: -x[1])
        texts.append(", ".join(c.lower() for c, _ in hits[:max_findings]) if hits else "none")
    df["prompt"] = texts
    return df


def build_prompt_text(prompt):
    return "findings: %s. report:" % (prompt if prompt else "none")


class CXRReportDataset(Dataset):
    def __init__(self, df, tf, train=True):
        self.paths = df["local_path"].tolist()
        self.reports = df["report"].tolist()
        self.prompts = df["prompt"].tolist()
        self.tf = tf
        self.train = train

    def __len__(self):
        return len(self.paths)

    def _image(self, i):
        for k in range(10):
            j = (i + k) % len(self.paths)
            try:
                img = Image.open(self.paths[j]).convert("RGB")
            except Exception:
                continue
            return self.tf(img), j
        raise RuntimeError("too many unreadable images near index %d" % i)

    def __getitem__(self, i):
        tok = get_tokenizer()
        img, j = self._image(i)

        p_ids = tok(build_prompt_text(self.prompts[j]), add_special_tokens=False)["input_ids"]
        r_ids = tok(" " + self.reports[j].strip(), add_special_tokens=False)["input_ids"]
        r_ids = r_ids[:MAX_REPORT_TOKENS] + [tok.eos_token_id]

        ids = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids
        return {
            "img": img,
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "prompt_ids": torch.tensor(p_ids),
            "ref": self.reports[j].strip(),
        }


def collate_train(batch):
    tok = get_tokenizer()
    pad = tok.pad_token_id
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = b["input_ids"]
        lab[i, :L] = b["labels"]
        att[i, :L] = 1
    return {
        "img": torch.stack([b["img"] for b in batch]),
        "input_ids": ids,
        "labels": lab,
        "attention_mask": att,
    }


def collate_gen(batch):
    """Left padding, because generation continues from the right-hand end."""
    tok = get_tokenizer()
    pad = tok.pad_token_id
    n = max(len(b["prompt_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["prompt_ids"])
        ids[i, n - L:] = b["prompt_ids"]
        att[i, n - L:] = 1
    return {
        "img": torch.stack([b["img"] for b in batch]),
        "input_ids": ids,
        "attention_mask": att,
        "refs": [b["ref"] for b in batch],
    }


def load_splits(data_dir=".", use_oracle=False):
    df = load_pairs(data_dir)
    df = attach_prompts(df, data_dir, use_oracle=use_oracle)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "validate"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    if len(va) == 0:
        va = te
    return tr, va, te