# MICCAI ViT-B MAE — AD Classification on T1 MRI

Binary classification of Alzheimer's Disease (AD vs CN) using a 3D Vision Transformer
pretrained with Masked Autoencoder (MAE) on BraTS+IXI+OASIS3, fine-tuned on ADNI T1 data.

Based on: "A ViT Recipe for AD Classification" (MICCAI 2024)

## Results

Test AUROC: ~0.64 +/- 0.05 (mean over multiple runs, 108 test subjects)
Best single run: 0.70

## Setup

### 1. Data

The T1 data should be at `data/T1/ADNI/` with one folder per patient (e.g. `003_S_0908/`).
Each patient folder contains one `.nii` file: a spatially normalized, skull-stripped,
N3-corrected T1 image from ADNI.

The train/val/test split CSV should be at `data/splits/t1_split.csv`.

### 2. Checkpoints (download from Google Drive)

Download and place in the `pretrained/` directory:

- `pretrained/miccai_vit_mae_75.pth` (472MB) — MAE pretrained encoder weights.
  Needed for training from scratch.
- `results/miccai_vit_t1/best_model.pt` (339MB) — Our best fine-tuned checkpoint.
  Needed for inference or as a starting point for multimodal integration.

### 3. Dependencies

```bash
pip install torch monai einops scikit-learn nibabel numpy scipy
```

### 4. Path Configuration

Two path constants at the top of `run_miccai_t1.py` and `preprocess_t1.py` need to
match your environment:

```python
# In run_miccai_t1.py (lines 14-17):
BASE_DIR = Path("/home/fnp23/python_projects/neuro_interp")
PREPROC_DIR = BASE_DIR / "data" / "T1_preprocessed_miccai"
SPLIT_PATH = BASE_DIR / "data" / "splits" / "t1_split.csv"

# In preprocess_t1.py (lines at top):
BASE_DIR = "/home/fnp23/python_projects/neuro_interp"
```

Change `/home/fnp23/python_projects/neuro_interp` to your project root.
On RunPod this would typically be `/root`.

## Preprocessing

Run once to create preprocessed `.npy` files from raw NIfTI:

```bash
python preprocess_t1.py
```

This does: Reorient LAS->RAS, z-score normalize (nonzero voxels), resize to 128x128x128,
save as float32 `.npy` with shape `(1, 128, 128, 128)`.

Output goes to `data/T1_preprocessed_miccai/`.

## Training

### Recommended run (reproduces best results):

```bash
python run_miccai_t1.py \
    --no_freeze \
    --batch_size 4 \
    --num_workers 2 \
    --lr_finetune 1e-4 \
    --weight_decay 0.3 \
    --phase2_epochs 50 \
    --patience 15
```

### All arguments:

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `pretrained/miccai_vit_mae_75.pth` | Path to MAE pretrained weights |
| `--batch_size` | 4 | Batch size. 4 fits on 24GB GPU, up to 16 on 80GB A100 |
| `--num_workers` | 4 | Dataloader workers. Use 0 if you get multiprocessing errors |
| `--phase1_epochs` | 20 | Phase 1 epochs (frozen backbone, head only). Ignored with `--no_freeze` |
| `--phase2_epochs` | 40 | Phase 2 epochs (or total epochs with `--no_freeze`) |
| `--lr_head` | 1e-3 | Phase 1 learning rate. Ignored with `--no_freeze` |
| `--lr_finetune` | 5e-5 | Phase 2 / no-freeze learning rate |
| `--weight_decay` | 0.05 | AdamW weight decay. Use 0.3 with `--no_freeze` |
| `--patience` | 15 | Early stopping patience (epochs without val AUROC improvement) |
| `--no_freeze` | False | Skip Phase 1, unfreeze all params from start with flat LR + warmup |
| `--output_dir` | `results/miccai_vit_t1` | Where to save checkpoints and logs |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

### Training modes:

**`--no_freeze` (recommended):** All 88.6M params unfrozen from epoch 0. Uses linear
warmup (10 epochs) + cosine decay. Single flat LR for all layers. This produced our
best results (~0.64-0.70 test AUROC).

**Two-phase (default, without `--no_freeze`):**
- Phase 1: Backbone frozen, only classification head trains (1,538 params).
- Phase 2: All params unfrozen with layerwise LR decay (0.9 per block).
- Generally worse than `--no_freeze` on this dataset due to the frozen features not
  being linearly separable for AD/CN.

## Loading the fine-tuned checkpoint for inference

```python
import torch
from miccai_vit.model import MICCAIViTClassifier

model = MICCAIViTClassifier(n_classes=2)
ckpt = torch.load("results/miccai_vit_t1/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Input: (batch, 1, 128, 128, 128) float32 tensor, z-scored per volume
# Output: dict with "logits" (batch, 2) and "features" (batch, 768)
```

## Data details

- 214 labeled ADNI patients (106 train split, 108 test split)
- Train split further divided: 84 train / 22 val (stratified)
- Input: T1-weighted MRI, spatially normalized to MNI, skull-stripped, N3-corrected
- Shape: 110x110x110 at 2mm isotropic, resized to 128x128x128
- Class distribution: ~52% AD / 48% CN (train), ~61% AD / 39% CN (test)
- Labels from: `data/project_1_3_data/IID/ADNI_binary_training.csv` and `..._testing.csv`

## Known limitations

- 84 training samples causes high variance between runs (test AUROC ranges 0.59-0.70)
- 22 validation samples makes checkpoint selection unreliable
- Model memorizes training set within ~10-15 epochs
- Results should be reported as mean +/- std over multiple runs
