import json
import os
import re
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224
MAX_LEN = 100
PROMPT_LEN = 16
MIN_FREQ = 3

PAD, BOS, EOS, UNK, SEP = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"]

LABEL_COLS = [
    "No Finding",
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]

_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _NORM,
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    _NORM,
])


def tokenize(text):
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9%./\- ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().split()


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len=MAX_LEN):
        ids = [self.stoi.get(t, UNK) for t in tokens][: max_len - 1] + [EOS]
        return ids + [PAD] * (max_len - len(ids))

    def decode(self, ids):
        words = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i > SEP:
                words.append(self.itos[i])
        return words

    def save(self, path):
        json.dump(self.itos, open(path, "w"))

    @staticmethod
    def load(path):
        return Vocab(json.load(open(path)))

    @staticmethod
    def build(texts, min_freq=MIN_FREQ):
        c = Counter()
        for t in texts:
            c.update(tokenize(t))
        for lab in LABEL_COLS:
            c.update(tokenize(lab))
        words = sorted([w for w, n in c.items() if n >= min_freq])
        return Vocab(SPECIALS + words), c


def make_prompt_ids(vocab, findings, prompt_len=PROMPT_LEN):
    ids = [BOS]
    for f in findings:
        ids += [vocab.stoi.get(w, UNK) for w in tokenize(f)]
    ids = ids[: prompt_len - 1]
    ids.append(SEP)
    return ids + [PAD] * (prompt_len - len(ids))


class PromptedReportDataset(Dataset):
    def __init__(self, df, vocab, tf, max_len=MAX_LEN, prompt_len=PROMPT_LEN):
        self.paths = df["local_path"].tolist()
        self.reports = df["report"].tolist()
        self.prompts = df["prompt_findings"].tolist()
        self.vocab = vocab
        self.tf = tf
        self.max_len = max_len
        self.prompt_len = prompt_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        for k in range(10):
            j = (i + k) % len(self.paths)
            try:
                img = Image.open(self.paths[j]).convert("RGB")
            except Exception:
                continue
            findings = [f for f in str(self.prompts[j]).split("|") if f]
            pid = make_prompt_ids(self.vocab, findings, self.prompt_len)
            rid = self.vocab.encode(tokenize(self.reports[j]), self.max_len)
            return (self.tf(img),
                    torch.tensor(pid, dtype=torch.long),
                    torch.tensor(rid, dtype=torch.long))
        raise RuntimeError("too many unreadable images near index %d" % i)


def attach_prompts(df, probs, thresholds, use_oracle=False, max_findings=3):
    if use_oracle:
        def row_findings(r):
            pos = [c for c in LABEL_COLS[1:] if r.get(c, 0) == 1]
            return "|".join(pos) if pos else "No Finding"
        df = df.copy()
        df["prompt_findings"] = df.apply(row_findings, axis=1)
        return df

    df = df.merge(probs, on="dicom_id", how="left")
    pcols = [f"p_{c}" for c in LABEL_COLS]
    for c in pcols:
        df[c] = df[c].fillna(0.0)

    def row_findings(r):
        hits = [(c, r[f"p_{c}"]) for c in LABEL_COLS[1:]
                if r[f"p_{c}"] >= thresholds.get(c, 0.5)]
        hits.sort(key=lambda x: -x[1])
        names = [c for c, _ in hits[:max_findings]]
        return "|".join(names) if names else "No Finding"

    df["prompt_findings"] = df.apply(row_findings, axis=1)
    return df


def load_pairs(data_dir=".", use_oracle=False, one_image_per_study=True,
               val_fraction=0.05, seed=42):
    pairs = pd.read_csv(os.path.join(data_dir, "image_report_pairs.csv"))
    avail = pd.read_csv(os.path.join(data_dir, "available_labels.csv"))

    df = pairs.merge(avail[["dicom_id", "local_path"]], on="dicom_id", how="inner")
    df = df[df["report"].notna()]
    if one_image_per_study:
        df = df.sort_values("dicom_id").drop_duplicates(subset=["study_id"])
    df = df.reset_index(drop=True)

    probs = pd.read_csv(os.path.join(data_dir, "prompt_probs.csv"))
    thresholds = json.load(open(os.path.join(data_dir, "prompt_thresholds.json")))
    df = attach_prompts(df, probs, thresholds, use_oracle=use_oracle)

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validate"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    if len(val) < 500:
        extra = train.sample(n=int(len(train) * val_fraction), random_state=seed)
        train = train.drop(extra.index).reset_index(drop=True)
        val = pd.concat([val, extra], ignore_index=True)

    return train, val, test
