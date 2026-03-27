# Multimodal Alzheimer's Disease Classification

Multimodal AD classification using 3D Vision Transformers and language models, with gradient-based interpretability analysis.
Coursework for **L349 (Machine Learning for the Physical World)**, University of Cambridge.

## Project Structure

```
miccai_vit/              # 3D ViT-B MAE encoder for T1 MRI
multimodal/              # Late-fusion pipeline (embedding extraction, MLP, training)
text_baselines/          # ModernBERT text encoder, SAE, TEO interpretability
preprocessing/           # T1 MRI preprocessing (reorient, normalize, resize)
notebooks/               # Data analysis and XAI visualizations
tests/                   # Unit tests
data/                    # Splits and clinical text CSVs
results/                 # Figures and evaluation outputs
run_miccai_t1.py         # Vision baseline training script
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Checkpoints

Model checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1nBqVgt6NNANREnfX8xBxfr2CZ2cDnPl2?usp=share_link).

Download and place them in the project root preserving the folder structure:

```
pretrained/
  miccai_vit_mae_75.pth              # MAE pretrained weights (for retraining)
results/
  miccai_vit_t1/
    best_model.pt                    # Fine-tuned 3D ViT-B on T1 MRI
  multimodal/
    fusion_model_seed42.pt           # Trained multimodal fusion MLP
llm_checkpoints/
  modernbert_adni_finetuned/
    config.json                      # ModernBERT config
    model.safetensors                # Fine-tuned ModernBERT weights
```

## Running

### 1. Extract multimodal embeddings (requires GPU)
```bash
python multimodal/extract_embeddings.py
```

### 2. Train and evaluate the fusion model
```bash
python multimodal/run_fusion.py
```

### 3. Interpretability analysis
Open and run `notebooks/xai_multimodal.ipynb`.

## Authors

- Umer Hasan (suh25@cam.ac.uk)
- Oszkar Urban (ou222@cam.ac.uk)
- Foivos Papathanasiou (fnp23@cam.ac.uk)
