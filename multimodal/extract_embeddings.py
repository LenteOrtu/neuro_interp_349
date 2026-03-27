"""
Extract frozen embeddings from the vision (MICCAI ViT) and text (ModernBERT) encoders.

Saves a single .pt file with all 214 patient embeddings for fast fusion training.
Requires GPU for vision encoder forward pass.
"""

import csv
import os
import sys
import logging
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MANIFEST = os.path.join(BASE_DIR, "data", "splits", "multimodal_manifest.csv")
VISION_CKPT = os.path.join(BASE_DIR, "results", "miccai_vit_t1", "best_model.pt")
TEXT_CKPT = os.path.join(BASE_DIR, "llm_checkpoints", "modernbert_adni_finetuned")
TOKENIZER_NAME = "answerdotai/ModernBERT-base"
OUTPUT = os.path.join(BASE_DIR, "data", "embeddings", "all_embeddings.pt")


def load_manifest():
    records = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            records.append(row)
    return records


def extract_vision_embeddings(records, device):
    """Extract 768-dim CLS embeddings from frozen MICCAI ViT."""
    from miccai_vit.model import MICCAIViTClassifier

    log.info("Loading vision model...")
    model = MICCAIViTClassifier(n_classes=2).to(device)
    ckpt = torch.load(VISION_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Vision model loaded (epoch {ckpt['epoch']}, val AUROC {ckpt['auroc']:.4f})")

    embeddings = {}
    with torch.no_grad():
        for rec in tqdm(records, desc="Vision embeddings"):
            pid = int(rec["patient_id"])
            arr = np.load(rec["npy_path"])  # (1, 128, 128, 128)
            img = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 1, 128, 128, 128)
            out = model(img)
            embeddings[pid] = out["features"].squeeze(0).cpu()  # (768,)

    log.info(f"Extracted {len(embeddings)} vision embeddings")
    return embeddings


def extract_text_embeddings(records, device):
    """Extract 768-dim mean-pooled embeddings from frozen ModernBERT."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    log.info("Loading text model...")
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_CKPT).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    log.info("Text model loaded")

    embeddings = {}
    with torch.no_grad():
        for rec in tqdm(records, desc="Text embeddings"):
            pid = int(rec["patient_id"])
            text = rec["generated_text"]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Mean pooling over non-padding tokens (matches classifier_pooling="mean")
            h = outputs.hidden_states[22]  # (1, seq_len, 768)
            mask = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
            text_emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # (1, 768)

            embeddings[pid] = text_emb.squeeze(0).cpu()  # (768,)

    log.info(f"Extracted {len(embeddings)} text embeddings")
    return embeddings


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    records = load_manifest()
    log.info(f"Manifest: {len(records)} patients")

    # Extract
    v_embs = extract_vision_embeddings(records, device)
    t_embs = extract_text_embeddings(records, device)

    # Combine into single dict
    all_embeddings = {}
    for rec in records:
        pid = int(rec["patient_id"])
        all_embeddings[pid] = {
            "v_cls": v_embs[pid],
            "t_emb": t_embs[pid],
            "label": int(rec["label"]),
            "split": rec["split"],
        }

    # Verify
    n_nan = sum(1 for v in all_embeddings.values()
                if torch.isnan(v["v_cls"]).any() or torch.isnan(v["t_emb"]).any())
    log.info(f"NaN check: {n_nan} patients with NaN (should be 0)")

    sample = list(all_embeddings.values())[0]
    log.info(f"Sample shapes: v_cls={sample['v_cls'].shape}, t_emb={sample['t_emb'].shape}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    torch.save(all_embeddings, OUTPUT)
    log.info(f"Saved {len(all_embeddings)} embeddings to {OUTPUT}")


if __name__ == "__main__":
    main()
