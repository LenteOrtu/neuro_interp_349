#!/usr/bin/env python3
"""
Train MICCAI 2024 ViT-B for AD classification on T1 MRI.

Uses preprocessed T1 .npy files (MNI-registered, skull-stripped, N3-corrected).

Usage:
    python run_miccai_t1.py
    python run_miccai_t1.py --batch_size 4 --epochs 50 --lr 1e-4
"""

import argparse
import csv
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader

from miccai_vit.model import MICCAIViTClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
PREPROC_DIR = BASE_DIR / "data" / "T1_preprocessed_miccai"
CSV_DIR = BASE_DIR / "data" / "project_1_3_data" / "IID"
SPLIT_PATH = BASE_DIR / "data" / "splits" / "t1_split.csv"
EXCLUDE_PIDS = set()


# ── Dataset ──────────────────────────────────────────────────────────────


class T1DatasetNpy(Dataset):
    """Loads preprocessed 128³ .npy T1 volumes with optional augmentation."""

    def __init__(self, records, preproc_dir, augment=False):
        self.records = records
        self.preproc_dir = Path(preproc_dir)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        npy_path = self.preproc_dir / f"{rec['pid']}.npy"
        try:
            arr = np.load(npy_path)  # (1, 128, 128, 128)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load preprocessed MICCAI volume for PID {rec['pid']} from {npy_path}"
            ) from e
        img = torch.from_numpy(arr)  # already float32 with channel dim

        if self.augment:
            # NO spatial augmentation — MNI-registered data with position-dependent
            # pretrained features. Flips and rotations would break spatial priors.
            # Only mild intensity jitter:
            if torch.rand(1).item() < 0.15:
                factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # [0.9, 1.1]
                img = img * factor
            if torch.rand(1).item() < 0.15:
                offset = (torch.rand(1).item() - 0.5) * 0.2  # [-0.1, 0.1]
                img = img + offset

        label = torch.tensor(rec["label"], dtype=torch.long)
        return {"image": img, "label": label, "pid": rec["pid"]}


# ── Data helpers ─────────────────────────────────────────────────────────


def load_labels(csv_path):
    """Load CSV → {patient_id (int): label (int)}."""
    labels = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            labels[int(row["Patient_ID"])] = int(float(row["Label"]))
    return labels


def is_valid_preprocessed_file(path):
    """Return True when a preprocessed .npy exists and can be memory-mapped."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        np.load(path, mmap_mode="r")
    except Exception:
        return False
    return True


def load_split(split_path, preproc_dir):
    """Load T1 train/val/test split and return record lists."""
    split_data = {"train": [], "val": [], "test": []}
    preproc_dir = Path(preproc_dir)
    skipped_invalid = 0
    with open(split_path) as f:
        for row in csv.DictReader(f):
            pid = int(row["patient_id"])
            split = row["split"]
            label = int(row["label"])
            if pid in EXCLUDE_PIDS:
                continue
            if not is_valid_preprocessed_file(preproc_dir / f"{pid}.npy"):
                skipped_invalid += 1
                continue
            split_data[split].append({
                "label": label,
                "pid": pid,
            })
    for s in split_data:
        log.info(f"  {s}: {len(split_data[s])} samples")
    if skipped_invalid:
        log.warning(f"Skipped {skipped_invalid} missing/corrupt preprocessed file(s)")
    return split_data


# ── Training ─────────────────────────────────────────────────────────────


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(images)
            loss = criterion(out["logits"], labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    use_amp = device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long()

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(images)
            loss = criterion(out["logits"], labels)

        probs = torch.softmax(out["logits"], dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()
        n_batches += 1

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "auroc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        "bal_acc": balanced_accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "sens": (preds[all_labels == 1].sum() / max(all_labels.sum(), 1)),
        "spec": ((1 - preds[all_labels == 0]).sum() / max((1 - all_labels).sum(), 1)),
    }
    return metrics, all_probs, all_labels


def find_optimal_threshold(probs, labels):
    """Two-stage threshold search on validation data."""
    best_score, best_t = -1, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        score = balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_t = t
    # Fine refinement
    for t in np.arange(max(0.01, best_t - 0.02), min(0.99, best_t + 0.02), 0.001):
        preds = (probs >= t).astype(int)
        score = balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="pretrained/miccai_vit_mae_75.pth")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--phase1_epochs", type=int, default=20)
    parser.add_argument("--phase2_epochs", type=int, default=40)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output_dir", default="results/miccai_vit_t1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_freeze", action="store_true",
                        help="Skip Phase 1, unfreeze all params from start with flat LR")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)} "
                 f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ── Data ──
    log.info("Loading split...")
    split_data = load_split(SPLIT_PATH, PREPROC_DIR)

    train_ds = T1DatasetNpy(split_data["train"], PREPROC_DIR, augment=True)
    val_ds = T1DatasetNpy(split_data["val"], PREPROC_DIR, augment=False)
    test_ds = T1DatasetNpy(split_data["test"], PREPROC_DIR, augment=False)

    # Class counts for weighted loss
    train_label_list = [d["label"] for d in split_data["train"]]
    class_counts = Counter(train_label_list)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    log.info(f"Class distribution (train): {dict(class_counts)}")

    # ── Model ──
    log.info("Building model...")
    model = MICCAIViTClassifier(n_classes=2).to(device)
    model.load_pretrained(str(BASE_DIR / args.checkpoint))

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {total_params:,}")

    # Class-weighted loss
    n_samples = len(train_label_list)
    class_weights = torch.tensor(
        [n_samples / (2 * class_counts[c]) for c in sorted(class_counts)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    log.info(f"Class weights: {class_weights.tolist()}")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_auroc = float("-inf")
    patience_counter = 0

    if args.no_freeze:
        # ══════════════════════════════════════════════════════════════
        # No-freeze mode: all params unfrozen, flat LR, linear warmup + cosine decay
        # (Reproduces the original 0.6468 AUROC run)
        # ══════════════════════════════════════════════════════════════
        total_epochs = args.phase2_epochs
        log.info(f"=== No-freeze mode: all params unfrozen ({total_epochs} epochs, lr={args.lr_finetune}) ===")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Trainable params: {trainable:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr_finetune,
            weight_decay=args.weight_decay,
        )

        # Linear warmup for 10 epochs, then cosine decay
        warmup_epochs = 10
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-7
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        for epoch in range(total_epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            log.info(
                f"Epoch {epoch:3d}/{total_epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                f"AUROC={val_metrics['auroc']:.4f} bal_acc={val_metrics['bal_acc']:.4f} "
                f"sens={val_metrics['sens']:.4f} spec={val_metrics['spec']:.4f} "
                f"F1={val_metrics['f1']:.4f} lr={lr:.2e} ({elapsed:.1f}s)"
            )

            if val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                patience_counter = 0
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "auroc": best_auroc,
                }
                torch.save(ckpt, os.path.join(args.output_dir, "best_model.pt"))
                log.info(f"Saved checkpoint (AUROC={best_auroc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    log.info(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break

    else:
        total_epochs = args.phase1_epochs + args.phase2_epochs

        # ══════════════════════════════════════════════════════════════
        # Phase 1: Frozen backbone — train head only
        # ══════════════════════════════════════════════════════════════
        log.info(f"=== Phase 1: Head only ({args.phase1_epochs} epochs) ===")
        model.freeze_backbone()
        trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Trainable params: {trainable_p1:,}")

        optimizer_p1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        )
        scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_p1, T_max=args.phase1_epochs, eta_min=1e-5
        )

        for epoch in range(args.phase1_epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer_p1, scaler, device)
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler_p1.step()
            lr = optimizer_p1.param_groups[0]["lr"]
            elapsed = time.time() - t0

            log.info(
                f"Epoch {epoch:3d}/{total_epochs} [P1] "
                f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                f"AUROC={val_metrics['auroc']:.4f} bal_acc={val_metrics['bal_acc']:.4f} "
                f"sens={val_metrics['sens']:.4f} spec={val_metrics['spec']:.4f} "
                f"F1={val_metrics['f1']:.4f} lr={lr:.2e} ({elapsed:.1f}s)"
            )

            if val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                patience_counter = 0
                ckpt = {
                    "epoch": epoch, "phase": 1,
                    "model_state_dict": model.state_dict(),
                    "auroc": best_auroc,
                }
                torch.save(ckpt, os.path.join(args.output_dir, "best_model.pt"))
                log.info(f"Saved checkpoint (AUROC={best_auroc:.4f})")
            else:
                patience_counter += 1

        # ══════════════════════════════════════════════════════════════
        # Phase 2: Unfreeze backbone — fine-tune with lower LR
        # ══════════════════════════════════════════════════════════════
        log.info(f"=== Phase 2: Full fine-tune ({args.phase2_epochs} epochs, lr={args.lr_finetune}) ===")
        model.unfreeze_all()
        trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Trainable params: {trainable_p2:,}")

        # Layerwise LR decay (0.9 per layer, matching MICCAI paper)
        param_groups = []
        # Head gets higher LR
        param_groups.append({
            "params": list(model.head.parameters()),
            "lr": args.lr_finetune * 10,  # head at 10x backbone LR
        })
        # Backbone layers with decay
        if hasattr(model, "blocks"):
            n_layers = len(model.blocks)
            for i, block in enumerate(model.blocks):
                layer_lr = args.lr_finetune * (0.9 ** (n_layers - 1 - i))
                param_groups.append({
                    "params": list(block.parameters()),
                    "lr": layer_lr,
                })
            # Patch embedding and other backbone params at lowest LR
            other_params = []
            block_params = set()
            for block in model.blocks:
                block_params.update(id(p) for p in block.parameters())
            head_params = {id(p) for p in model.head.parameters()}
            for p in model.parameters():
                if id(p) not in block_params and id(p) not in head_params:
                    other_params.append(p)
            if other_params:
                param_groups.append({
                    "params": other_params,
                    "lr": args.lr_finetune * (0.9 ** n_layers),
                })
        else:
            param_groups.append({
                "params": [p for p in model.parameters() if id(p) not in {id(hp) for hp in model.head.parameters()}],
                "lr": args.lr_finetune,
            })

        optimizer_p2 = torch.optim.AdamW(
            param_groups,
            weight_decay=args.weight_decay,
        )
        scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_p2, T_max=args.phase2_epochs, eta_min=1e-7
        )

        # Reset patience for phase 2 but keep best_auroc
        patience_counter = 0

        for epoch in range(args.phase2_epochs):
            global_epoch = args.phase1_epochs + epoch
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer_p2, scaler, device)
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler_p2.step()
            lr = optimizer_p2.param_groups[0]["lr"]
            elapsed = time.time() - t0

            log.info(
                f"Epoch {global_epoch:3d}/{total_epochs} [P2] "
                f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                f"AUROC={val_metrics['auroc']:.4f} bal_acc={val_metrics['bal_acc']:.4f} "
                f"sens={val_metrics['sens']:.4f} spec={val_metrics['spec']:.4f} "
                f"F1={val_metrics['f1']:.4f} lr={lr:.2e} ({elapsed:.1f}s)"
            )

            if val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                patience_counter = 0
                ckpt = {
                    "epoch": global_epoch, "phase": 2,
                    "model_state_dict": model.state_dict(),
                    "auroc": best_auroc,
                }
                torch.save(ckpt, os.path.join(args.output_dir, "best_model.pt"))
                log.info(f"Saved checkpoint (AUROC={best_auroc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    log.info(f"Early stopping at epoch {global_epoch} (no improvement for {args.patience} epochs)")
                    break

    # ── Test evaluation ──
    log.info("=" * 60)
    log.info("FINAL TEST EVALUATION")
    log.info("=" * 60)

    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    log.info(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (AUROC={best_ckpt['auroc']:.4f})")

    # Find optimal threshold on validation
    val_metrics, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
    opt_threshold, opt_bal_acc = find_optimal_threshold(val_probs, val_labels)
    log.info(f"Optimal threshold on val: {opt_threshold:.3f} (bal_acc={opt_bal_acc:.4f})")

    # Test with default and optimal thresholds
    test_metrics, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    test_preds_opt = (test_probs >= opt_threshold).astype(int)

    log.info(f"Test (t=0.50): AUROC={test_metrics['auroc']:.4f} "
             f"bal_acc={test_metrics['bal_acc']:.4f} "
             f"sens={test_metrics['sens']:.4f} spec={test_metrics['spec']:.4f}")
    log.info(f"Test (t={opt_threshold:.2f}): AUROC={test_metrics['auroc']:.4f} "
             f"bal_acc={balanced_accuracy_score(test_labels, test_preds_opt):.4f} "
             f"sens={test_preds_opt[test_labels == 1].mean():.4f} "
             f"spec={(1 - test_preds_opt[test_labels == 0]).mean():.4f}")

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"Best val AUROC: {best_auroc:.4f} (epoch {best_ckpt['epoch']})")
    log.info(f"Test AUROC: {test_metrics['auroc']:.4f}")


if __name__ == "__main__":
    main()
