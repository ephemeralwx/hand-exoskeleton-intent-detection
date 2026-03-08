import os
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

MAT_FILE = "labeled data 4 subjects.mat"
OUTPUT_DIR = "tcn_loso_results"

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60
PATIENCE = 12

# arbitrary arch based on trial and error
CHANNELS = [32, 64, 64, 64]
KERNEL_SIZE = 5
DROPOUT = 0.25

THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()

                # same-ish padding for odd ker size. breaks on even but whatever we only use 5
        padding = ((kernel_size - 1) * dilation) // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

                # off-by-1 from padding edge cases. just crop to match
        if out.size(-1) != residual.size(-1):
            min_len = min(out.size(-1), residual.size(-1))
            out = out[..., :min_len]
            residual = residual[..., :min_len]

        return self.final_relu(out + residual)

class ResidualTCN(nn.Module):
    def __init__(self, input_channels: int, channels: List[int], kernel_size: int, dropout: float):
        super().__init__()

        blocks = []
        in_ch = input_channels
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            blocks.append(
                ResidualTCNBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x).squeeze(-1)
        return logits

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(yb.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets).astype(int)
    metrics = compute_metrics(y_true, y_prob, threshold=THRESHOLD)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        running_loss += loss.item() * xb.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(yb.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets).astype(int)
    metrics = compute_metrics(y_true, y_prob, threshold=THRESHOLD)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics, y_true, y_prob

def _is_numeric_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)

def _squeeze_label_array(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    y = np.squeeze(y)
    if y.ndim != 1:
        raise ValueError(f"Labels should become 1D after squeeze, got shape {y.shape}")
    y = y.astype(np.int64)
    uniq = np.unique(y)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"Labels are not binary 0/1. Found unique values: {uniq}")
    return y

def _try_format_windows(X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    
    X = np.asarray(X)
    y = _squeeze_label_array(y)
    n = len(y)

    if X.ndim == 2:
        if X.shape[0] == n:
            feat = X.shape[1]
            if feat == 800:
                return X.reshape(n, 100, 8).astype(np.float32)
            return None

    if X.ndim != 3:
        return None

    shape = X.shape

        # matlab loads dims in random order. just brute force perms to find N first
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]

    best = None
    for p in perms:
        Xt = np.transpose(X, p)
        if Xt.shape[0] != n:
            continue

        T, C = Xt.shape[1], Xt.shape[2]
        if 20 <= T <= 500 and 2 <= C <= 64:
                        # heuristic to find 100x8 windows in matlab spaghetti
            score = abs(T - 100) + abs(C - 8)
            if best is None or score < best[0]:
                best = (score, Xt.astype(np.float32))

    if best is not None:
        return best[1]

    return None

def _walk_python_object(obj: Any, path: str = "root") -> List[Tuple[str, np.ndarray, np.ndarray]]:
    
    found = []

    if isinstance(obj, dict):
        keys = list(obj.keys())

        possible_x_keys = ["X", "x", "data", "windows", "signals", "features"]
        possible_y_keys = ["Y", "y", "labels", "label", "target", "targets"]

        x_candidates = [k for k in keys if k in possible_x_keys]
        y_candidates = [k for k in keys if k in possible_y_keys]

        for xk in x_candidates:
            for yk in y_candidates:
                try:
                    Xf = _try_format_windows(obj[xk], obj[yk])
                    if Xf is not None:
                        y = _squeeze_label_array(obj[yk])
                        found.append((f"{path}.{xk}/{yk}", Xf, y))
                except Exception:
                    # silently ignore broken structs cos matlab is weird
                    pass

        for k, v in obj.items():
            if k.startswith("__"):
                continue
            found.extend(_walk_python_object(v, f"{path}.{k}"))
        return found

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            found.extend(_walk_python_object(item, f"{path}[{i}]"))
        return found

    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        try:
            if obj.size == 1:
                scalar = obj.item()
                if isinstance(scalar, tuple) and obj.dtype.names is not None:
                    struct_dict = {name: scalar[i] for i, name in enumerate(obj.dtype.names)}
                    found.extend(_walk_python_object(struct_dict, path))
                else:
                    for name in obj.dtype.names:
                        found.extend(_walk_python_object(obj[name], f"{path}.{name}"))
            else:
                for idx in np.ndindex(obj.shape):
                    elem = obj[idx]
                    if isinstance(elem, tuple):
                        struct_dict = {name: elem[i] for i, name in enumerate(obj.dtype.names)}
                        found.extend(_walk_python_object(struct_dict, f"{path}{idx}"))
        except Exception:
            pass
        return found

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        try:
            for idx, item in np.ndenumerate(obj):
                found.extend(_walk_python_object(item, f"{path}{idx}"))
        except Exception:
            pass
        return found

    return found

def load_mat_scipy(path: str):
    from scipy.io import loadmat
    return loadmat(path, squeeze_me=True, struct_as_record=False)

def load_mat_h5(path: str):
    import h5py

    def convert(obj):
        if isinstance(obj, h5py.Dataset):
            arr = obj[()]
            if isinstance(arr, np.ndarray) and arr.dtype.kind in ("u", "i") and arr.ndim >= 1:
                return arr
            return arr

        if isinstance(obj, h5py.Group):
            out = {}
            for k in obj.keys():
                out[k] = convert(obj[k])
            return out

        return obj

    with h5py.File(path, "r") as f:
        data = {}
        for k in f.keys():
            data[k] = convert(f[k])
        return data

def deduplicate_candidates(cands: List[Tuple[str, np.ndarray, np.ndarray]]):
    unique = []
    seen = set()
    for path, X, y in cands:
        key = (X.shape, y.shape, int(y.sum()))
        if key not in seen:
            seen.add(key)
            unique.append((path, X, y))
    return unique

def group_candidates_into_subjects(cands: List[Tuple[str, np.ndarray, np.ndarray]]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    
    good = []
    for path, X, y in cands:
        if X.ndim == 3 and y.ndim == 1 and X.shape[0] == len(y):
            if X.shape[0] >= 100 and 20 <= X.shape[1] <= 500 and 2 <= X.shape[2] <= 64:
                good.append((path, X, y))

    if len(good) == 0:
        raise RuntimeError("No candidate subject datasets found in the .mat file.")

    good.sort(key=lambda item: (abs(item[1].shape[1] - 100) + abs(item[1].shape[2] - 8), -item[1].shape[0]))

    if len(good) == 4:
        return good

    if len(good) > 4:
        selected = []
        used_shapes = set()
        for item in good:
            shape_key = item[1].shape
            if shape_key not in used_shapes:
                selected.append(item)
                used_shapes.add(shape_key)
            if len(selected) == 4:
                break
        if len(selected) < 4:
            selected = good[:4]
        return selected

    return good

def load_subjects_from_mat(mat_path: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    obj = None
    scipy_error = None
    try:
        obj = load_mat_scipy(mat_path)
    except Exception as e:
        scipy_error = e

    candidates = []
    if obj is not None:
        if "data" in obj and "labels" in obj and isinstance(obj["data"], np.ndarray) and isinstance(obj["labels"], np.ndarray):
            data_arr = obj["data"]
            labels_arr = obj["labels"]
            if data_arr.shape == labels_arr.shape and data_arr.size > 0:
                for idx in range(data_arr.size):
                    X_subj = np.stack(data_arr.flat[idx]) if data_arr.flat[idx].dtype == object else data_arr.flat[idx]
                    y_subj = np.stack(labels_arr.flat[idx]) if labels_arr.flat[idx].dtype == object else labels_arr.flat[idx]
                    Xf = _try_format_windows(X_subj, y_subj)
                    if Xf is not None:
                        candidates.append((f"subject_{idx}", Xf, _squeeze_label_array(y_subj)))
        
        if len(candidates) == 0:
            candidates = _walk_python_object(obj, "mat")
        candidates = deduplicate_candidates(candidates)

    if len(candidates) == 0:
        try:
            obj_h5 = load_mat_h5(mat_path)
            candidates = _walk_python_object(obj_h5, "h5")
            candidates = deduplicate_candidates(candidates)
        except Exception as e:
            msg = f"Could not parse MAT file.\nscipy error: {scipy_error}\nh5 error: {e}"
            raise RuntimeError(msg)

    subjects = group_candidates_into_subjects(candidates)
 
    if len(subjects) < 2:
        raise RuntimeError(
            "Found too few subject datasets in the MAT file. "
            "The internal structure may need a small custom loader tweak."
        )

    return subjects

def fit_normalizer(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0
    return mean, std

def apply_normalizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MAT_FILE):
        raise FileNotFoundError(f"Could not find {MAT_FILE} in the current directory.")

    print("=" * 80)
    print("Loading MATLAB data...")
    print("=" * 80)

    subjects = load_subjects_from_mat(MAT_FILE)

    print(f"\nDetected {len(subjects)} subject dataset(s):")
    for i, (name, X, y) in enumerate(subjects):
        pos_rate = float(y.mean())
        print(
            f"  Subject {i + 1}: {name}\n"
            f"    X shape = {X.shape}, y shape = {y.shape}, "
            f"positive rate = {pos_rate:.4f}"
        )

    results = []

    for test_idx in range(len(subjects)):
        print("\n" + "=" * 80)
        print(f"LOSO Fold {test_idx + 1}/{len(subjects)}")
        print("=" * 80)

        test_name, X_test, y_test = subjects[test_idx]

        X_train_list = []
        y_train_list = []
        train_subject_names = []

        for j, (name, X, y) in enumerate(subjects):
            if j != test_idx:
                X_train_list.append(X)
                y_train_list.append(y)
                train_subject_names.append(name)

        X_train_full = np.concatenate(X_train_list, axis=0)
        y_train_full = np.concatenate(y_train_list, axis=0)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.15,
            random_state=SEED,
            stratify=y_train_full,
        )

        mean, std = fit_normalizer(X_train)
        X_train = apply_normalizer(X_train, mean, std)
        X_val = apply_normalizer(X_val, mean, std)
        X_test_norm = apply_normalizer(X_test, mean, std)

        print(f"Train subjects: {train_subject_names}")
        print(f"Test subject:   {test_name}")
        print(f"Train shape:    {X_train.shape}")
        print(f"Val shape:      {X_val.shape}")
        print(f"Test shape:     {X_test_norm.shape}")
        print(f"Train pos rate: {y_train.mean():.4f}")
        print(f"Val pos rate:   {y_val.mean():.4f}")
        print(f"Test pos rate:  {y_test.mean():.4f}")

        train_ds = WindowDataset(X_train, y_train)
        val_ds = WindowDataset(X_val, y_val)
        test_ds = WindowDataset(X_test_norm, y_test)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        input_channels = X_train.shape[2]
        model = ResidualTCN(
            input_channels=input_channels,
            channels=CHANNELS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
        ).to(DEVICE)

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
                # hacky bce weight to handle massive pos class imbalance
        pos_weight_val = float(neg_count / max(pos_count, 1.0))
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4
        )

        best_val_f1 = -1.0
        best_epoch = -1
        patience_counter = 0
        best_model_path = os.path.join(OUTPUT_DIR, f"best_tcn_fold_{test_idx + 1}.pt")

        history = []

        for epoch in range(1, EPOCHS + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)

            scheduler.step(val_metrics["f1"])
            history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d} | "
                f"LR {current_lr:.2e} | "
                f"Train Loss {train_metrics['loss']:.4f} | "
                f"Train F1 {train_metrics['f1']:.4f} | "
                f"Val Loss {val_metrics['loss']:.4f} | "
                f"Val Acc {val_metrics['accuracy']:.4f} | "
                f"Val Prec {val_metrics['precision']:.4f} | "
                f"Val Rec {val_metrics['recall']:.4f} | "
                f"Val F1 {val_metrics['f1']:.4f} | "
                f"Val ROC-AUC {val_metrics['roc_auc']:.4f}"
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
                patience_counter = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "mean": mean,
                        "std": std,
                        "input_channels": input_channels,
                        "channels": CHANNELS,
                        "kernel_size": KERNEL_SIZE,
                        "dropout": DROPOUT,
                        "threshold": THRESHOLD,
                        "test_subject_name": test_name,
                    },
                    best_model_path,
                )
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

        checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics, y_true_test, y_prob_test = evaluate(model, test_loader, criterion, DEVICE)

        print("\nBest validation epoch:", best_epoch)
        print("Final test metrics on held-out subject:")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        fold_result = {
            "fold": test_idx + 1,
            "test_subject_name": test_name,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics,
            "train_subjects": train_subject_names,
            "history": history,
        }
        results.append(fold_result)

        np.savez(
            os.path.join(OUTPUT_DIR, f"fold_{test_idx + 1}_test_outputs.npz"),
            y_true=y_true_test,
            y_prob=y_prob_test,
        )

        with open(os.path.join(OUTPUT_DIR, f"fold_{test_idx + 1}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    summary = {}
    for m in metric_names:
        vals = [r["test_metrics"][m] for r in results]
        summary[m] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "values": [float(v) for v in vals],
        }

    print("\n" + "=" * 80)
    print("Overall LOSO results summary:")
    print("=" * 80)
    for m in metric_names:
        print(
            f"{m:10s} | mean = {summary[m]['mean']:.4f} | "
            f"std = {summary[m]['std']:.4f} | values = {summary[m]['values']}"
        )

    with open(os.path.join(OUTPUT_DIR, "loso_summary.json"), "w") as f:
        json.dump(
            {
                "config": {
                    "MAT_FILE": MAT_FILE,
                    "BATCH_SIZE": BATCH_SIZE,
                    "LR": LR,
                    "WEIGHT_DECAY": WEIGHT_DECAY,
                    "EPOCHS": EPOCHS,
                    "PATIENCE": PATIENCE,
                    "CHANNELS": CHANNELS,
                    "KERNEL_SIZE": KERNEL_SIZE,
                    "DROPOUT": DROPOUT,
                    "THRESHOLD": THRESHOLD,
                    "DEVICE": DEVICE,
                },
                "summary": summary,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved all outputs to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()