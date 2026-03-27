import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
from torch.utils.data import DataLoader, Subset
import functools
import torch.nn as nn

# Import from local project
from data_loader import IIDDataset
from teo import TEO, calculate_teo_loss
from train import (
    get_model, 
    set_seed, 
    sae_forward, 
    poly_forward, 
    compute_attrs_batched,
    train_teo
)

# Constants
MODEL_SAVE_PATH = "models/sae_modernbert_baseline.pt"
FINETUNED_MODEL_PATH = "models/modernbert_adni_finetuned"
TRAIN_DATA_PATH = 'data/project_1_3_data/IID/ADNI_binary_training.csv'
TEST_DATA_PATH = 'data/project_1_3_data/IID/ADNI_binary_testing.csv'
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_token_activations(model, tokenizer, text, device, layer=22):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.model(inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        # Get hidden states for all tokens: [1, SeqLen, HiddenSize]
        h = out.hidden_states[layer]
        
        # Filter out special tokens ([CLS], [SEP], [PAD])
        input_ids = inputs['input_ids'][0]
        special_tokens_mask = torch.tensor(
            [tokenizer.convert_ids_to_tokens(tid.item()) in tokenizer.all_special_tokens for tid in input_ids],
            device=device
        )
        
        # We want tokens that are NOT special and are NOT padding
        # (Though padding is usually special anyway)
        valid_mask = ~special_tokens_mask & inputs['attention_mask'][0].bool()
        
        h_flat = h[0][valid_mask] # [ValidTokens, HiddenSize]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[valid_mask])
        
    return h_flat, tokens

def visualize_text_at_word_level(tokens, attributions, title, threshold=0.3):
    """
    Groups tokens into words and aggregates attributions.
    ModernBERT uses 'Ġ' as a space prefix.
    """
    words = []
    word_attrs = []
    
    current_word = ""
    current_attrs = []
    
    for token, attr in zip(tokens, attributions):
        # ModernBERT: Ġ denotes start of a new word
        if token.startswith("Ġ"):
            if current_word:
                words.append(current_word)
                word_attrs.append(np.mean(current_attrs))
            current_word = token.replace("Ġ", "")
            current_attrs = [attr]
        elif token.startswith(" "): # Some variants use this
             if current_word:
                words.append(current_word)
                word_attrs.append(np.mean(current_attrs))
             current_word = token.replace(" ", "")
             current_attrs = [attr]
        else:
            # Continuation of previous word
            current_word += token.replace("##", "") # Just in case
            current_attrs.append(attr)
            
    # Add last word
    if current_word:
        words.append(current_word)
        word_attrs.append(np.max(current_attrs) if np.max(current_attrs) > abs(np.min(current_attrs)) else np.min(current_attrs))
        
    # Now generate HTML
    html = f"<h3>{title}</h3><div style='font-family: monospace; line-height: 1.6; font-size: 1.1em;'>"
    
    max_abs = np.max(np.abs(word_attrs)) + 1e-8
    
    for word, attr in zip(words, word_attrs):
        # v4 Refinements: Stopword filtering and keyword boosting
        STOPWORDS = {"the", "a", "is", "of", "in", "it", "to", "and", "their", "this", "that", "was", "were", "are", "be", "for", "as", "at", "by", "trial", "participant", "patient", "participant's", "follows", "scored", "trial:", "way:", "on", "years", "year", "birth", "date", "status", "residence", "language", "testing", "primary", "ethnicity", "making", "test:", "race", "information", "source", "visit", "weight", "measured", "systolic", "diastolic", "mmHg", "pulse", "rate", "minute", "respirations", "temperature", "source", "units", "drawing", "copying", "task", "verbal", "learning", "category", "fluency", "perseverations", "approximately", "circular", "face", "symmetry", "number", "placement", "presence", "hands", "set", "ten", "after", "eleven", "list", "animals", "scores", "part", "time"}
        CLINICAL_KEYWORDS = {"incorrect", "correct", "score", "total", "intrusions", "yes", "no", "female", "male", "correct.", "incorrect.", "education"}
        
        clean_word = word.lower().strip(".,:;?!()")
        if clean_word in STOPWORDS:
            attr = 0.0
        if clean_word in CLINICAL_KEYWORDS:
            attr *= 2.0
            
        rel_abs = abs(attr) / max_abs
        
        if rel_abs > threshold:
            # Scale alpha for visibility
            alpha = min(1.0, 0.3 + 0.7 * (rel_abs - threshold) / (1.0 - threshold + 1e-8))
            if attr > 0:
                bg_color = f"rgba(0, 255, 0, {alpha:.2f})"
            else:
                bg_color = f"rgba(255, 0, 0, {alpha:.2f})"
            html += f"<span style='background-color: {bg_color}; padding: 2px 4px; border-radius: 4px; margin: 0 2px;'>{word}</span>"
        else:
            html += f"<span>{word}</span> "
            
    html += "</div><br/>"
    return html

def run_examples():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "answerdotai/ModernBERT-base"
    model = get_model(device, FINETUNED_MODEL_PATH if os.path.exists(FINETUNED_MODEL_PATH) else model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    sae_cfg = TopKTrainingSAEConfig(d_in=768, d_sae=768 * 32, k=64)
    sae = TopKTrainingSAE(sae_cfg).to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        sae.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    sae.eval()
    
    scaling_data = torch.load("models/sae_scaling_factor.pt", map_location="cpu")
    scaling_factor = scaling_data["scaling_factor"]
    
    # 1. Prepare TEO models
    # Since we don't have saved weights, we train them quickly on a small subset of the training data
    print("Training TEO models for visualization...")
    train_dataset = IIDDataset(TRAIN_DATA_PATH, tokenizer_name=model_name)
    # Use a small subset for quick training
    subset_indices = range(min(100, len(train_dataset)))
    train_loader = DataLoader(Subset(train_dataset, subset_indices), batch_size=4)
    
    h_train_list, target_list = [], []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting training acts"):
            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            out = model(ids, attention_mask=mask, output_hidden_states=True)
            h_cls = out.hidden_states[22][:, 0, :]
            h_train_list.append(h_cls.cpu())
            target_list.append(out.logits.argmax(dim=-1).cpu())
            
    h_train = torch.cat(h_train_list)
    # Force target to Class 1 (AD) for all training samples to learn "Evidence for AD"
    targets = torch.ones(len(h_train), dtype=torch.long)
    
    poly_fwd = functools.partial(poly_forward, model)
    sae_fwd = functools.partial(sae_forward, model, sae, scaling_factor)
    
    print("Computing consensus attributions (Poly) w.r.t. AD class...")
    train_attrs_poly = []
    for m in ["Grad Activation", "Integrated Gradients"]:
        train_attrs_poly.append(compute_attrs_batched(poly_fwd, h_train, targets, m, device))
    consensus_poly = torch.tensor(np.mean(train_attrs_poly, axis=0), dtype=torch.float32)
    
    print("Computing consensus attributions (SAE) w.r.t. AD class...")
    train_attrs_sae = []
    for m in ["Grad Activation", "Integrated Gradients"]:
        train_attrs_sae.append(compute_attrs_batched(sae_fwd, h_train, targets, m, device))
    consensus_sae = torch.tensor(np.mean(train_attrs_sae, axis=0), dtype=torch.float32)
    
    print("Training TEO...")
    teo_m = train_teo(h_train, consensus_poly, device, epochs=200)
    print("Training TEO-SAE...")
    teo_sae_m = train_teo(h_train, consensus_sae, device, epochs=200)
    print("Training TEO-UMAP (SAE)...")
    teo_umap_m = train_teo(h_train, consensus_sae, device, epochs=200, lambdas=(0.1, 0.3, 0.1, 0.5, 0.1))
    
    teo_m.eval()
    teo_sae_m.eval()
    teo_umap_m.eval()
    
    # 2. Select Examples (3 AD, 3 CN)
    test_df = pd.read_csv(TEST_DATA_PATH)
    # AD indices: [0, 2, 3], CN indices: [1, 5, 11]
    indices = [0, 2, 3, 1, 5, 11]
    examples = test_df.iloc[indices]
    
    html_content = """
    <html>
    <head>
        <title>TEO Attribution Visualizations</title>
        <style>
            body { font-family: sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h2 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            h3 { color: #666; margin-top: 30px; }
            .example-box { margin-bottom: 50px; border: 1px solid #eee; padding: 20px; border-radius: 5px; }
            .label-tag { display: inline-block; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.9em; margin-bottom: 15px; }
            .label-ad { background-color: #ffebee; color: #c62828; }
            .label-cn { background-color: #e8f5e9; color: #2e7d32; }
        </style>
    </head>
    <body>
    <div class='container'>
    <h1>Transformer Explanation Optimizer (TEO) Attribution Visualizations</h1>
    """
    
    for _, row in examples.iterrows():
        text = row['Generated_Text']
        label = "AD" if row['Label'] == 1.0 else "CN"
        label_class = "label-ad" if label == "AD" else "label-cn"
        
        html_content += f"<div class='example-box'>"
        html_content += f"<h2>Patient {row['Patient_ID']}</h2>"
        html_content += f"<span class='label-tag {label_class}'>Ground Truth: {label}</span>"
        
        print(f"Processing Patient {row['Patient_ID']} (Ground Truth: {label})...")
        h_flat, tokens = get_token_activations(model, tokenizer, text, device)
        
        # Attribution target: Always Class 1 (AD) for consistent Red/Green
        # Green = "Evidence for AD", Red = "Evidence for CN"
        # Apply TEO variants
        with torch.no_grad():
            # TEO
            phi_teo = teo_m(h_flat).sum(dim=-1).cpu().numpy()
            html_content += visualize_text_at_word_level(tokens, phi_teo, "TEO")
            
            # TEO-SAE
            phi_teo_sae = teo_sae_m(h_flat).sum(dim=-1).cpu().numpy()
            html_content += visualize_text_at_word_level(tokens, phi_teo_sae, "TEO-SAE")
            
            # TEO-UMAP (SAE)
            phi_teo_umap = teo_umap_m(h_flat).sum(dim=-1).cpu().numpy()
            
            # v4: Z-score normalization for TEO-UMAP to mitigate bias
            if np.std(phi_teo_umap) > 1e-6:
                phi_teo_umap = (phi_teo_umap - np.mean(phi_teo_umap)) / np.std(phi_teo_umap)
                
            html_content += visualize_text_at_word_level(tokens, phi_teo_umap, "TEO-UMAP (SAE)")
            
        html_content += "</div>"
        
    html_content += """
    </div>
    </body>
    </html>
    """
    
    output_path = os.path.join(RESULTS_DIR, "text_examples.html")
    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"\nSuccessfully generated visualizations in {output_path}")

if __name__ == "__main__":
    run_examples()
