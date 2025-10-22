# utils.py
import os
from collections import Counter
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RACE_MAP = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Other"
}

AGE_GROUPS = {
    "child": (0, 17),
    "young_adult": (18, 39),
    "middle_age": (40, 64),
    "senior": (65, 200),
}

MAX_AGE = 116
ALLOWED_GENDERS = {"0", "1"}
ALLOWED_RACES = set(str(k) for k in RACE_MAP.keys())


def parse_utk_filename(path):
    base = os.path.basename(path)
    if "__" in base:
        raise ValueError("double_underscore")

    name, ext = os.path.splitext(base)
    tokens = name.split("_")
    if len(tokens) != 4:
        raise ValueError(f"wrong_token_count:{len(tokens)}")

    age_t, gender_t, race_t, ts_t = tokens

    if not age_t.isdigit():
        raise ValueError("age_not_digits")
    age = int(age_t)
    if not (0 <= age <= MAX_AGE):
        raise ValueError(f"age_out_of_range:{age}")

    if not gender_t.isdigit() or gender_t not in ALLOWED_GENDERS:
        raise ValueError(f"gender_invalid:{gender_t}")
    gender = int(gender_t)

    if not race_t.isdigit() or race_t not in ALLOWED_RACES:
        raise ValueError(f"race_invalid:{race_t}")
    race = int(race_t)

    if not ts_t.isdigit():
        raise ValueError("timestamp_not_digits")

    return {"age": age, "gender": gender, "race": race, "file": path, "basename": base}


def age_to_group(age):
    for name, (lo, hi) in AGE_GROUPS.items():
        if lo <= age <= hi:
            return name
    return "unknown"


# Dataset
class UTKFaceDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        self.meta = [parse_utk_filename(f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.meta[idx]["gender"])
        return img, label, self.meta[idx]



def get_transforms(train=True):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.Resize((224, 224)),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            normalize,
        ])


def build_model(num_classes=2, pretrained=True, device="cuda"):
    try:
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


# Class weights for CrossEntropyLoss
def compute_class_weights(labels):
    cnt = Counter(labels)
    if len(cnt) == 0:
        return torch.tensor([1.0], dtype=torch.float)
    max_label = max(cnt.keys())
    weights = [1.0] * (max_label + 1)
    total = sum(cnt.values())
    num_present = len(cnt)
    for lbl, c in cnt.items():
        weights[lbl] = total / (num_present * c)
    return torch.tensor(weights, dtype=torch.float)



# Evaluate loop helper
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []
    metas = []

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs = batch[0]
            labels = batch[1]
            meta = batch[2] if len(batch) > 2 else None
        else:
            raise RuntimeError("Unexpected batch format from loader")

        imgs = imgs.to(device)
        out = model(imgs)
        probs = torch.softmax(out, dim=1)
        pred = probs.argmax(dim=1).cpu().numpy()
        preds.extend(pred.tolist())
        trues.extend(labels.cpu().numpy().tolist())

        if meta is None:
            continue

        if isinstance(meta, dict):
            first_key = next(iter(meta))
            batch_len = len(meta[first_key])
            for i in range(batch_len):
                sample_meta = {k: v[i] for k, v in meta.items()}
                metas.append(sample_meta)
        elif isinstance(meta, (list, tuple)):
            metas.extend(meta)
        else:
            metas.append(meta)

    return np.array(trues), np.array(preds), metas


# Subgroup metrics by overall, age and race (as mentioned in the assignment)
def subgroup_metrics(trues, preds, metas):

    overall = {
        "accuracy": float(accuracy_score(trues, preds)),
        "precision": float(precision_score(trues, preds, zero_division=0)),
        "recall": float(recall_score(trues, preds, zero_division=0)),
        "f1": float(f1_score(trues, preds, zero_division=0))
    }

    by_age = {}
    for ag_name in AGE_GROUPS.keys():
        idxs = [i for i, m in enumerate(metas) if age_to_group(m["age"]) == ag_name]
        if not idxs:
            continue
        y_t = trues[idxs]
        y_p = preds[idxs]
        by_age[ag_name] = {
            "n": len(idxs),
            "accuracy": float(accuracy_score(y_t, y_p)),
            "precision": float(precision_score(y_t, y_p, zero_division=0)),
            "recall": float(recall_score(y_t, y_p, zero_division=0)),
            "f1": float(f1_score(y_t, y_p, zero_division=0))
        }

    by_race = {}
    for rcode, rname in RACE_MAP.items():
        idxs = [i for i, m in enumerate(metas) if m["race"] == rcode]
        if not idxs:
            continue
        y_t = trues[idxs]
        y_p = preds[idxs]
        by_race[rname] = {
            "n": len(idxs),
            "accuracy": float(accuracy_score(y_t, y_p)),
            "precision": float(precision_score(y_t, y_p, zero_division=0)),
            "recall": float(recall_score(y_t, y_p, zero_division=0)),
            "f1": float(f1_score(y_t, y_p, zero_division=0))
        }

    return {"overall": overall, "by_age": by_age, "by_race": by_race}
