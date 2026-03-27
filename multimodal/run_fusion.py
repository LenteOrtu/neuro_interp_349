"""
Multimodal fusion training: MLPFusion on cached vision + text embeddings.

Phase A: 5-fold CV on 106 dev patients to validate recipe
Phase B: Retrain once on all 106 dev patients, evaluate on 108 test patients
Also runs unimodal linear probe baselines for comparison.
"""

import os
import sys
import logging
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from multimodal.models import MLPFusion

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "all_embeddings.pt")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "splits", "multimodal_manifest.csv")


# ── Helpers ─────────────────────────────────────────────────────────────


def load_embeddings():
    """Load cached embeddings and split into dev (train+val) and test."""
    all_emb = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=False)

    dev_v, dev_t, dev_y, dev_pids = [], [], [], []
    test_v, test_t, test_y, test_pids = [], [], [], []

    for pid, emb in all_emb.items():
        v = emb["v_cls"]
        t = emb["t_emb"]
        label = emb["label"]
        split = emb["split"]

        if split in ("train", "val"):
            dev_v.append(v)
            dev_t.append(t)
            dev_y.append(label)
            dev_pids.append(pid)
        else:
            test_v.append(v)
            test_t.append(t)
            test_y.append(label)
            test_pids.append(pid)

    return {
        "dev": {
            "v": torch.stack(dev_v), "t": torch.stack(dev_t),
            "y": torch.tensor(dev_y, dtype=torch.long), "pids": dev_pids,
        },
        "test": {
            "v": torch.stack(test_v), "t": torch.stack(test_t),
            "y": torch.tensor(test_y, dtype=torch.long), "pids": test_pids,
        },
    }


def load_manifest_text():
    """Load generated text keyed by patient ID from multimodal manifest."""
    text_by_pid = {}
    with open(MANIFEST_PATH) as f:
        for row in csv.DictReader(f):
            text_by_pid[int(row["patient_id"])] = row["generated_text"]
    return text_by_pid


def compute_class_weights(labels):
    """Inverse frequency class weights."""
    counts = np.bincount(labels.numpy())
    total = len(labels)
    weights = total / (len(counts) * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def find_optimal_threshold(probs, labels):
    """Two-stage threshold search on validation data."""
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        score = balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_t = t
    # Fine search
    for t in np.arange(max(0.01, best_t - 0.02), min(0.99, best_t + 0.02), 0.001):
        preds = (probs >= t).astype(int)
        score = balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


def evaluate(probs, labels, threshold=0.5):
    """Compute metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)
    auroc = roc_auc_score(labels, probs)
    bal_acc = balanced_accuracy_score(labels, preds)
    sens = preds[labels == 1].mean() if (labels == 1).any() else 0.0
    spec = 1.0 - preds[labels == 0].mean() if (labels == 0).any() else 0.0
    f1 = f1_score(labels, preds, zero_division=0)
    return {"auroc": auroc, "bal_acc": bal_acc, "sens": sens, "spec": spec, "f1": f1}


# ── Training ────────────────────────────────────────────────────────────


def train_model(model, train_v, train_t, train_y, val_v, val_t, val_y,
                epochs=200, lr=1e-3, weight_decay=0.1, patience=30,
                label_smoothing=0.1, batch_size=16):
    """Train fusion model, return best model state and metrics."""

    device = next(model.parameters()).device
    class_weights = compute_class_weights(train_y).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    dataset = TensorDataset(train_v, train_t, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_auroc = float("-inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for v_batch, t_batch, y_batch in loader:
            v_batch = v_batch.to(device)
            t_batch = t_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out = model(v_batch, t_batch)
            loss = criterion(out["logits"], y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(val_v.to(device), val_t.to(device))
            val_probs = torch.softmax(val_out["logits"], dim=1)[:, 1].cpu().numpy()
            val_labels = val_y.numpy()

        try:
            val_auroc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auroc = 0.5

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Convert the zero-based index into an actual epoch count.
    return best_state, best_auroc, best_epoch + 1


def run_phase_a(data, args):
    """5-fold CV on 106 dev patients."""
    log.info("=" * 60)
    log.info("PHASE A: 5-fold CV on dev patients")
    log.info("=" * 60)

    dev = data["dev"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_aurocs = []
    fold_thresholds = []
    fold_best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(dev["v"], dev["y"])):
        log.info(f"\n--- Fold {fold + 1}/5 ---")

        train_v, val_v = dev["v"][train_idx], dev["v"][val_idx]
        train_t, val_t = dev["t"][train_idx], dev["t"][val_idx]
        train_y, val_y = dev["y"][train_idx], dev["y"][val_idx]

        model = MLPFusion(dropout=args.dropout).to(args.device)
        best_state, best_auroc, best_epoch = train_model(
            model, train_v, train_t, train_y, val_v, val_t, val_y,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            patience=args.patience, batch_size=args.batch_size,
        )

        # Load best state and find threshold
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_out = model(val_v.to(args.device), val_t.to(args.device))
            val_probs = torch.softmax(val_out["logits"], dim=1)[:, 1].cpu().numpy()

        threshold, _ = find_optimal_threshold(val_probs, val_y.numpy())

        fold_aurocs.append(best_auroc)
        fold_thresholds.append(threshold)
        fold_best_epochs.append(best_epoch)

        log.info(f"  Fold {fold + 1}: AUROC={best_auroc:.4f}, threshold={threshold:.3f}, best_epochs={best_epoch}")

    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    median_threshold = np.median(fold_thresholds)
    avg_best_epoch = max(1, int(round(np.mean(fold_best_epochs))))

    log.info(f"\nPhase A summary:")
    log.info(f"  Val AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    log.info(f"  Median threshold: {median_threshold:.3f}")
    log.info(f"  Avg best epoch count: {avg_best_epoch}")

    return median_threshold, avg_best_epoch


def run_phase_b(data, args, threshold, n_epochs):
    """Train final model on all 106 dev patients, evaluate on 108 test patients."""
    log.info("\n" + "=" * 60)
    log.info("PHASE B: Final model on all dev patients")
    log.info("=" * 60)

    dev = data["dev"]
    test = data["test"]

    model = MLPFusion(dropout=args.dropout).to(args.device)

    # Train on all dev data — no validation, fixed epochs
    device = args.device
    class_weights = compute_class_weights(dev["y"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    dataset = TensorDataset(dev["v"], dev["t"], dev["y"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        for v_batch, t_batch, y_batch in loader:
            v_batch, t_batch, y_batch = v_batch.to(device), t_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(v_batch, t_batch)
            loss = criterion(out["logits"], y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        test_out = model(test["v"].to(device), test["t"].to(device))
        test_probs = torch.softmax(test_out["logits"], dim=1)[:, 1].cpu().numpy()
        test_labels = test["y"].numpy()

    metrics_default = evaluate(test_probs, test_labels, threshold=0.5)
    metrics_tuned = evaluate(test_probs, test_labels, threshold=threshold)

    log.info(f"\nTest results (t=0.50): AUROC={metrics_default['auroc']:.4f} bal_acc={metrics_default['bal_acc']:.4f} "
             f"sens={metrics_default['sens']:.4f} spec={metrics_default['spec']:.4f} F1={metrics_default['f1']:.4f}")
    log.info(f"Test results (t={threshold:.2f}): AUROC={metrics_tuned['auroc']:.4f} bal_acc={metrics_tuned['bal_acc']:.4f} "
             f"sens={metrics_tuned['sens']:.4f} spec={metrics_tuned['spec']:.4f} F1={metrics_tuned['f1']:.4f}")

    # Save model
    save_dir = os.path.join(BASE_DIR, "results", "multimodal")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "fusion_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "threshold": threshold,
        "n_epochs": n_epochs,
        "test_auroc": metrics_default["auroc"],
    }, save_path)
    log.info(f"Saved fusion model to {save_path}")

    return metrics_default, model


def run_baselines(data, args, threshold):
    """Run unimodal baselines for comparison."""
    log.info("\n" + "=" * 60)
    log.info("BASELINES")
    log.info("=" * 60)

    dev = data["dev"]
    test = data["test"]
    device = args.device
    text_by_pid = load_manifest_text()

    # 1. Vision-only linear probe
    log.info("\n--- Vision-only linear probe ---")
    v_model = nn.Linear(768, 2).to(device)
    v_optimizer = AdamW(v_model.parameters(), lr=1e-3, weight_decay=0.1)
    class_weights = compute_class_weights(dev["y"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    for epoch in range(200):
        v_model.train()
        v_optimizer.zero_grad()
        logits = v_model(dev["v"].to(device))
        loss = criterion(logits, dev["y"].to(device))
        loss.backward()
        v_optimizer.step()

    v_model.eval()
    with torch.no_grad():
        v_probs = torch.softmax(v_model(test["v"].to(device)), dim=1)[:, 1].cpu().numpy()
    v_metrics = evaluate(v_probs, test["y"].numpy(), threshold=0.5)
    log.info(f"  Vision probe: AUROC={v_metrics['auroc']:.4f} bal_acc={v_metrics['bal_acc']:.4f}")

    # 2. Text-only linear probe
    log.info("\n--- Text-only linear probe ---")
    t_model = nn.Linear(768, 2).to(device)
    t_optimizer = AdamW(t_model.parameters(), lr=1e-3, weight_decay=0.1)

    for epoch in range(200):
        t_model.train()
        t_optimizer.zero_grad()
        logits = t_model(dev["t"].to(device))
        loss = criterion(logits, dev["y"].to(device))
        loss.backward()
        t_optimizer.step()

    t_model.eval()
    with torch.no_grad():
        t_probs = torch.softmax(t_model(test["t"].to(device)), dim=1)[:, 1].cpu().numpy()
    t_metrics = evaluate(t_probs, test["y"].numpy(), threshold=0.5)
    log.info(f"  Text probe: AUROC={t_metrics['auroc']:.4f} bal_acc={t_metrics['bal_acc']:.4f}")

    # 3. Original ModernBERT logits
    log.info("\n--- Original ModernBERT (own head) ---")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    text_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(BASE_DIR, "llm_checkpoints", "modernbert_adni_finetuned")
    ).to(device)
    text_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    text_model_probs = []
    with torch.no_grad():
        for pid in test["pids"]:
            text = text_by_pid[pid]
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = text_model(**inputs)
            prob = torch.softmax(out.logits, dim=1)[:, 1].item()
            text_model_probs.append(prob)
    text_model_probs = np.array(text_model_probs)
    text_model_metrics = evaluate(text_model_probs, test["y"].numpy(), threshold=0.5)
    log.info(f"  ModernBERT: AUROC={text_model_metrics['auroc']:.4f} bal_acc={text_model_metrics['bal_acc']:.4f}")

    # 4. Original MICCAI ViT logits
    log.info("\n--- Original MICCAI ViT (own head) ---")
    from miccai_vit.model import MICCAIViTClassifier
    vit = MICCAIViTClassifier(n_classes=2).to(device)
    ckpt = torch.load(os.path.join(BASE_DIR, "results", "miccai_vit_t1", "best_model.pt"),
                       map_location=device, weights_only=False)
    vit.load_state_dict(ckpt["model_state_dict"])
    vit.eval()

    vit_probs = []
    with torch.no_grad():
        for pid in test["pids"]:
            arr = np.load(os.path.join(BASE_DIR, "data", "T1_preprocessed_miccai", f"{pid}.npy"))
            img = torch.from_numpy(arr).unsqueeze(0).to(device)
            out = vit(img)
            prob = torch.softmax(out["logits"], dim=1)[:, 1].item()
            vit_probs.append(prob)
    vit_probs = np.array(vit_probs)
    vit_metrics = evaluate(vit_probs, test["y"].numpy(), threshold=0.5)
    log.info(f"  MICCAI ViT: AUROC={vit_metrics['auroc']:.4f} bal_acc={vit_metrics['bal_acc']:.4f}")

    return v_metrics, t_metrics, text_model_metrics, vit_metrics


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    log.info(f"Device: {args.device}")

    data = load_embeddings()
    log.info(f"Dev: {len(data['dev']['y'])} patients, Test: {len(data['test']['y'])} patients")

    # Phase A: CV
    median_threshold, avg_best_epoch = run_phase_a(data, args)

    # Phase B: Final model
    fusion_metrics, fusion_model = run_phase_b(data, args, median_threshold, avg_best_epoch)

    # Baselines
    v_metrics, t_metrics, text_model_metrics, vit_metrics = run_baselines(data, args, median_threshold)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("FINAL SUMMARY (all at t=0.50)")
    log.info("=" * 60)
    log.info(f"  MICCAI ViT (own head):    AUROC={vit_metrics['auroc']:.4f}")
    log.info(f"  ModernBERT (own head):    AUROC={text_model_metrics['auroc']:.4f}")
    log.info(f"  Vision linear probe:      AUROC={v_metrics['auroc']:.4f}")
    log.info(f"  Text linear probe:        AUROC={t_metrics['auroc']:.4f}")
    log.info(f"  MLPFusion (ours):         AUROC={fusion_metrics['auroc']:.4f}")


if __name__ == "__main__":
    main()
