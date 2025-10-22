import os
import json
import random
from glob import glob
from collections import Counter
from datetime import datetime

import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from utils import (
    parse_utk_filename,
    UTKFaceDataset,
    get_transforms,
    build_model,
    compute_class_weights,
    evaluate,
    subgroup_metrics,
    age_to_group
)

DEFAULT_CONFIG_PATH = "train_config.yaml"

def load_config(path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()

    data_dirs = cfg.get("data_dirs", [])
    if not data_dirs:
        raise ValueError("No data_dirs specified in config.yaml")

    out_dir = cfg.get("out_dir", "./train_output")
    os.makedirs(out_dir, exist_ok=True)

    # seeds
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if (torch.cuda.is_available()) else "cpu"

    # gather files from 3 directories
    candidates = []
    for d in data_dirs:
        if not os.path.isdir(d):
            print(f"Warning: data_dir not found or not a directory: {d} (skipping)")
            continue
        candidates.extend(sorted(glob(os.path.join(d, "*.jpg")) + glob(os.path.join(d, "*.png"))))

    # dedupe by basename
    seen = set()
    files = []
    for p in candidates:
        bn = os.path.basename(p)
        if bn in seen:
            continue
        seen.add(bn)
        files.append(p)

    good_files = []
    metas = []
    bad = []
    for p in files:
        try:
            m = parse_utk_filename(p)
            good_files.append(p)
            metas.append(m)
        except Exception as e:
            bad.append((os.path.basename(p), str(e)))

    # report and save bad filenames
    report_path = os.path.join(out_dir, "strict_filename_report.json")

    summary = {
        "total_scanned": len(files),
        "good": len(good_files),
        "bad": len(bad),
        "bad_examples": bad[:200]
    }
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    all_files = good_files
    metas = metas

    # compute simple distributions and save run metadata
    genders = [m["gender"] for m in metas]
    races = [m["race"] for m in metas]
    ages = [m["age"] for m in metas]
    age_groups = [age_to_group(a) for a in ages]

    gender_counts = dict(Counter(genders))
    race_counts = dict(Counter(races))
    agegroup_counts = dict(Counter(age_groups))

    run_meta = {
        "seed": seed,
        "n_total": len(all_files),
        "gender_counts": gender_counts,
        "race_counts": race_counts,
        "age_group_counts": agegroup_counts,
        "use_sampler": bool(cfg.get("use_sampler", False)),
        "use_class_weights": bool(cfg.get("use_class_weights", False))
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as fh:
        json.dump(run_meta, fh, indent=2)


    # split sizes and validation
    val_size = float(cfg.get("val_size", 0.1))
    test_size = float(cfg.get("test_size", 0.1))
    test_total = val_size + test_size


    # Try stratifying by combined key gender_agegroup_race -> fallback to age_race -> age -> random
    idxs = list(range(len(all_files)))
    combined_keys = [f"{g}_{ag}_{r}" for g, ag, r in zip(genders, age_groups, races)]
    age_race_keys = [f"{ag}_{r}" for ag, r in zip(age_groups, races)]
    age_only_keys = [str(ag) for ag in age_groups]

    # gender_race_keys = [f"{g}_{r}" for g, r in zip(genders, races)]
    # gender_only_keys = [str(g) for g in genders]

    strat_mode = None
    try:
        if min(Counter(combined_keys).values()) >= 2:
            train_idx, temp_idx = train_test_split(idxs, test_size=test_total, random_state=seed, stratify=combined_keys)
            strat_mode = "gender_agegroup_race"
        # else:
        #     raise ValueError("combined strata sparse")
    except Exception:
        try:
            if min(Counter(age_race_keys).values()) >= 2:
                train_idx, temp_idx = train_test_split(idxs, test_size=test_total, random_state=seed, stratify=age_race_keys)
                strat_mode = "age_race"
            # else:
            #     raise ValueError("age_race sparse")
        except Exception:
            try:
                if min(Counter(age_only_keys).values()) >= 2:
                    train_idx, temp_idx = train_test_split(idxs, test_size=test_total, random_state=seed, stratify=age_only_keys)
                    strat_mode = "age_only"
                # else:
                #     raise ValueError("age_only sparse")
            except Exception:
                train_idx, temp_idx = train_test_split(idxs, test_size=test_total, random_state=seed, shuffle=True)
                strat_mode = "none"

    relative_test = test_size / test_total
    if strat_mode == "gender_agegroup_race":
        temp_strat = [combined_keys[i] for i in temp_idx]
    elif strat_mode == "age_race":
        temp_strat = [age_race_keys[i] for i in temp_idx]
    elif strat_mode == "age_only":
        temp_strat = [age_only_keys[i] for i in temp_idx]
    else:
        temp_strat = None

    if temp_strat is not None:
        try:
            val_idx_rel, test_idx_rel = train_test_split(list(range(len(temp_idx))), test_size=relative_test, random_state=seed, stratify=temp_strat)
        except Exception:
            val_idx_rel, test_idx_rel = train_test_split(list(range(len(temp_idx))), test_size=relative_test, random_state=seed, shuffle=True)
    else:
        val_idx_rel, test_idx_rel = train_test_split(list(range(len(temp_idx))), test_size=relative_test, random_state=seed, shuffle=True)

    val_idx = [temp_idx[i] for i in val_idx_rel]
    test_idx = [temp_idx[i] for i in test_idx_rel]

    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    test_files = [all_files[i] for i in test_idx]

    print(f"Split mode: {strat_mode}. Sizes -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # save split distribution
    def counts(indices):
        gs = [genders[i] for i in indices]
        rs = [races[i] for i in indices]
        ags = [age_groups[i] for i in indices]
        return {"n": len(indices), "gender_counts": dict(Counter(gs)), "race_counts": dict(Counter(rs)), "age_group_counts": dict(Counter(ags))}

    split_info = {"strat_mode": strat_mode, "total": {"n": len(all_files), "gender_counts": gender_counts, "race_counts": race_counts, "age_group_counts": agegroup_counts}, "train": counts(train_idx), "val": counts(val_idx), "test": counts(test_idx)}
    with open(os.path.join(out_dir, "split_distribution.json"), "w") as fh:
        json.dump(split_info, fh, indent=2)

    # datasets and loaders
    train_ds = UTKFaceDataset(train_files, transform=get_transforms(train=True))
    val_ds = UTKFaceDataset(val_files, transform=get_transforms(train=False))
    test_ds = UTKFaceDataset(test_files, transform=get_transforms(train=False))

    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 4))

    if cfg.get("use_sampler", False):
        train_labels = [m["gender"] for m in train_ds.meta]
        class_counts = Counter(train_labels)
        class_weights_map = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights_map[l] for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # build model and loss
    model = build_model(num_classes=2, pretrained=bool(cfg.get("pretrained", True)), device=device)
    class_weights = compute_class_weights([m["gender"] for m in metas]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if cfg.get("use_class_weights", False) else nn.CrossEntropyLoss()

    lr = float(cfg.get("lr", 0.01))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.get("lr_step", 7)), gamma=float(cfg.get("lr_gamma", 0.1)))

    best_val_acc = 0.0
    epochs = int(cfg.get("epochs", 12))
    save_every = int(cfg.get("save_every", 5))

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, labels, _meta in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
            if steps % 20 == 0:
                pbar.set_postfix(loss=running_loss / steps)

        scheduler.step()

        # validation
        y_true, y_pred, val_meta = evaluate(model, val_loader, device)
        val_acc = accuracy_score(y_true, y_pred)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}: TrainLoss={running_loss/steps:.4f}, ValAcc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(out_dir, "best_resnet18_gender.pth")
            torch.save({"epoch": epoch, 
                        "model_state": model.state_dict(), 
                        "optimizer_state": optimizer.state_dict(), 
                        "val_acc": val_acc}, ckpt)
            print(f"Saved best model -> {ckpt}")

        if epoch % save_every == 0:
            ckpt_epoch = os.path.join(out_dir, f"resnet18_epoch{epoch}.pth")
            torch.save({"epoch": epoch, 
                        "model_state": model.state_dict(), 
                        "optimizer_state": optimizer.state_dict(), 
                        "val_acc": val_acc}, ckpt_epoch)

    # load best and evaluate on test
    best_ckpt_path = os.path.join(out_dir, "best_resnet18_gender.pth")
    if os.path.exists(best_ckpt_path):
        state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        print(f"Loaded best checkpoint from {best_ckpt_path} (val_acc={state.get('val_acc')})")

    y_true, y_pred, test_meta = evaluate(model, test_loader, device)
    report = subgroup_metrics(y_true, y_pred, test_meta)

    # Create results file
    results_file = os.path.join(out_dir, "test_results.txt")
    
    def print_and_write(text, file_handle=None):
        print(text)
        if file_handle:
            file_handle.write(text + "\n")
    
    with open(results_file, "w") as f:
        print_and_write("\n Overall Test metrics: ", f)
        for k,v in report["overall"].items():
            print_and_write(f"{k}: {v:.4f}", f)

        print_and_write("\n Test metrics by age group: ", f)
        for ag, stats in report["by_age"].items():
            print_and_write(f"{ag}: n={stats['n']}, acc={stats['accuracy']:.4f}, f1={stats['f1']:.4f}", f)

        print_and_write("\n Test metrics by race: ", f)
        for rn, stats in report["by_race"].items():
            print_and_write(f"{rn}: n={stats['n']}, acc={stats['accuracy']:.4f}, f1={stats['f1']:.4f}", f)

        print_and_write("\nFull classification report (test):", f)
        classification_report_text = classification_report(y_true, y_pred, target_names=["male","female"], zero_division=0)
        print_and_write(classification_report_text, f)

    # save predictions CSV
    csv_out = os.path.join(out_dir, "test_predictions.csv")
    with open(csv_out, "w", newline="") as fh:
        import csv
        writer = csv.writer(fh)
        writer.writerow(["basename", "filepath", "age", "gender_true", "race", "pred_gender"])
        for i, m in enumerate(test_meta):
            writer.writerow([m["basename"], m["file"], m["age"], int(y_true[i]), m["race"], int(y_pred[i])])
    print(f"Saved test predictions -> {csv_out}")

if __name__ == "__main__":
    main()
