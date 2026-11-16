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
MIN_FREQ = 3

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

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
        ids = [BOS] + [self.stoi.get(t, UNK) for t in tokens][: max_len - 2] + [EOS]
        return ids + [PAD] * (max_len - len(ids))

    def decode(self, ids):
        words = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i > UNK:
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
        words = sorted([w for w, n in c.items() if n >= min_freq])
        return Vocab(SPECIALS + words), c


class ReportDataset(Dataset):
    def __init__(self, df, vocab, tf, max_len=MAX_LEN):
        self.paths = df["local_path"].tolist()
        self.reports = df["report"].tolist()
        self.vocab = vocab
        self.tf = tf
        self.max_len = max_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        for k in range(10):
            j = (i + k) % len(self.paths)
            try:
                img = Image.open(self.paths[j]).convert("RGB")
            except Exception:
                continue
            ids = self.vocab.encode(tokenize(self.reports[j]), self.max_len)
            return self.tf(img), torch.tensor(ids, dtype=torch.long)
        raise RuntimeError("too many unreadable images near index %d" % i)


def load_pairs(data_dir=".", one_image_per_study=True, val_fraction=0.05, seed=42):
    pairs = pd.read_csv(os.path.join(data_dir, "image_report_pairs.csv"))
    avail = pd.read_csv(os.path.join(data_dir, "available_labels.csv"))

    df = pairs.merge(avail[["dicom_id", "local_path"]], on="dicom_id", how="inner")
    df = df[df["report"].notna()]

    if one_image_per_study:
        df = df.sort_values("dicom_id").drop_duplicates(subset=["study_id"])

    df = df.reset_index(drop=True)

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validate"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    if len(val) < 500:
        extra = train.sample(n=int(len(train) * val_fraction), random_state=seed)
        train = train.drop(extra.index).reset_index(drop=True)
        val = pd.concat([val, extra], ignore_index=True)

    return train, val, test
