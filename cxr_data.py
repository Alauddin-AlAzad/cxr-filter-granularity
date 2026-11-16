"""
Data module for the chest X-ray classifier.

Changes from the earlier version:
  - images are padded to a square instead of centre cropped, so the costophrenic
    angles and the lateral chest wall are no longer cut off
  - input resolution raised from 224 to 320
  - horizontal flipping removed, since it mirrors thoracic anatomy and
    contradicts the laterality stated in the reports

Must stay an importable file, not notebook code, so DataLoader workers can
pickle the Dataset on Windows.
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 320

LABEL_COLS = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
]

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class PadSquare:
    """Pads the shorter side with black so the image becomes square, keeping
    the whole chest in frame and the aspect ratio unchanged."""

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        left = (side - w) // 2
        top = (side - h) // 2
        return ImageOps.expand(img, (left, top, side - w - left, side - h - top), fill=0)


train_tf = transforms.Compose([
    PadSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.97, 1.03)),
    transforms.ToTensor(),
    NORM,
])

eval_tf = transforms.Compose([
    PadSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORM,
])


class CXRDataset(Dataset):
    def __init__(self, df, tf):
        self.paths = df["local_path"].tolist()
        self.labels = df[LABEL_COLS].fillna(0).replace(-1, 0).values.astype("float32")
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
            return self.tf(img), torch.from_numpy(self.labels[j])
        raise RuntimeError("too many unreadable images near index %d" % i)


def load_frames(data_dir="."):
    df = pd.read_csv(os.path.join(data_dir, "available_labels.csv"))
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "validate"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    return tr, va, te


def pos_weight(df, lo=1.0, hi=20.0):
    y = df[LABEL_COLS].fillna(0).replace(-1, 0).values.astype("float32")
    pos = y.sum(axis=0)
    neg = len(y) - pos
    w = np.divide(neg, np.maximum(pos, 1.0))
    return torch.tensor(np.clip(w, lo, hi), dtype=torch.float32)