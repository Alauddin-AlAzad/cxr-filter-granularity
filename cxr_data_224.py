import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_COLS = [
    "No Finding",
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]

IMG_SIZE = 224
_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    _NORM,
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    _NORM,
])


class CXRDataset(Dataset):
    def __init__(self, df, tf):
        self.paths = df["local_path"].tolist()
        self.labels = df[LABEL_COLS].values.astype("float32")
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


def load_frames(data_dir=".", image_root="images", labels_csv="available_labels.csv",
                val_fraction=0.05, seed=42):
    df = pd.read_csv(os.path.join(data_dir, labels_csv))
    if "local_path" not in df.columns:
        df["local_path"] = df["image_path"].apply(
            lambda p: os.path.join(data_dir, image_root, p)
        )
    for c in LABEL_COLS:
        df[c] = df[c].fillna(0).replace(-1, 0).astype("float32")

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validate"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    if len(val) < 500:
        extra = train.sample(n=int(len(train) * val_fraction), random_state=seed)
        train = train.drop(extra.index).reset_index(drop=True)
        val = pd.concat([val, extra], ignore_index=True)

    return train, val, test
