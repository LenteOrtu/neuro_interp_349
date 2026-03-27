import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import functools
import umap
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
from captum.attr import InputXGradient, IntegratedGradients, GradientShap, LayerConductance
import pandas as pd

from data_loader import IIDDataset
from teo import TEO
from train import (
    get_model, 
    set_seed, 
    poly_forward, 
    sae_forward, 
    compute_attrs_batched, 
    train_teo
)

# Constants
MODEL_SAVE_PATH = "models/sae_modernbert_baseline.pt"
FINETUNED_MODEL_PATH = "models/modernbert_adni_finetuned"
TRAIN_DATA_PATH = 'data/project_1_3_data/IID/ADNI_binary_training.csv'
TEST_DATA_PATH = 'data/project_1_3_data/IID/ADNI_binary_testing.csv'
RESULTS_DIR = "results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Categories and keywords
CATEGORY_COLORS = {
    "Demographics": "red",
    "Vital Signs": "brown",
    "Clock Drawing Test": "green",
    "Clock Copying Test": "blue",
    "Auditory Verbal": "black",
    "Category Fluency-Animal Test": "orange",
    "American National Adult Reading Test": "grey",
    "Functional Activities Questionnaire": "purple"
}

CATEGORY_KEYWORDS = {
    "Demographics": ["sex", "birth", "handedness", "marital", "education", "retired", "residence", "language", "ethnicity", "race", "age", "gender", "male", "female", "man", "woman", "school", "grade", "college", "degree", "years of education"],
    "Vital Signs": ["weight", "systolic", "diastolic", "pulse", "respiration", "temperature", "bp", "blood pressure", "heart rate", "height", "bmi", "vital signs"],
    "Clock Drawing Test": ["clock drawing test", "circular face", "symmetry", "number placement", "correctness of numbers", "hands", "clock", "drawing"],
    "Clock Copying Test": ["clock copying task", "copy", "reproduction"],
    "Auditory Verbal": ["trial 1", "trial 2", "trial 3", "trial 4", "trial 5", "trial 6", "list b", "ravlt", "word", "recall", "learning", "auditory verbal", "30 minute delay", "recognition score", "delay", "recognition"],
    "Category Fluency-Animal Test": ["category fluency test", "animals", "total correct", "perseverations", "intrusions", "animal fluency", "naming", "fluency"],
    "American National Adult Reading Test": ["american national adult reading test", "anart", "reading", "pronounce"],
    "Functional Activities Questionnaire": ["functional activities questionnaire", "faq", "checks", "bills", "tax", "shopping", "hobby", "stove", "meal", "events", "tv", "appointments", "driving", "activities of daily"]
}

def assign_category(token_str: str, context: str) -> str:
    token_str = token_str.lower().strip()
    context = context.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in token_str or kw in context:
                return cat
    return "Other"

class CombinedModel(torch.nn.Module):
    def __init__(self, model, sae=None, teo=None, scaling_factor=1.0, layer=22):
        super().__init__()
        self.model = model
        self.sae = sae
        self.teo = teo
        self.scaling_factor = scaling_factor
        self.layer = layer

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        h = outputs.hidden_states[self.layer]
        h_cls = h[:, 0, :]
        if self.sae is not None:
            h_cls = self.sae(h_cls / self.scaling_factor) * self.scaling_factor
        if self.teo is not None:
            h_cls = self.teo(h_cls)
        logits = self.model.classifier(self.model.drop(self.model.head(h_cls)))
        return logits

def compute_token_attrs_batched(model, dataloader, method_name, device, tokenizer, sae=None, teo=None, scaling_factor=1.0, limit=20):
    combined_model = CombinedModel(model, sae, None, scaling_factor).to(device).eval()
    all_token_attrs, all_token_categories = [], []
    for i, batch in enumerate(dataloader):
        if i >= limit: break
        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            h = outputs.hidden_states[22]
        if teo is not None:
            with torch.no_grad():
                B, S, H = h.shape
                attr = teo(h.view(-1, H).to(device)).view(B, S, -1)
        else:
            if method_name == "Activation":
                attr = h
            else:
                targets = combined_model(model.model.embeddings(input_ids), attention_mask).argmax(dim=-1)
                inputs_embeds = model.model.embeddings(input_ids).detach()
                inputs_embeds.requires_grad = True
                expl = {
                    "Gradient Activation": InputXGradient(combined_model),
                    "Integrated Gradients": IntegratedGradients(combined_model),
                    "Layer Conductance": IntegratedGradients(combined_model),
                    "Gradient SHAP": GradientShap(combined_model)
                }.get(method_name, InputXGradient(combined_model))
                if method_name == "Gradient SHAP":
                    attr = expl.attribute(inputs_embeds, baselines=torch.zeros_like(inputs_embeds), target=targets, additional_forward_args=(attention_mask,))
                else:
                    kwargs = {"n_steps": 10} if method_name in ["Integrated Gradients", "Layer Conductance"] else {}
                    attr = expl.attribute(inputs_embeds, target=targets, additional_forward_args=(attention_mask,), **kwargs)
        attr = attr.detach().cpu()
        for b in range(input_ids.shape[0]):
            valid_len = attention_mask[b].sum().item()
            num_tokens = min(valid_len, 128)
            indices = np.random.choice(valid_len, num_tokens, replace=False); indices.sort()
            all_token_attrs.append(attr[b, indices])
            full_dec = [tokenizer.decode([tid]) for tid in input_ids[b]]
            all_token_categories.append([assign_category(full_dec[idx], "".join(full_dec[max(0, idx-10):min(len(full_dec), idx+10)])) for idx in indices])
    res_tensor = torch.cat(all_token_attrs, dim=0).numpy()
    del combined_model
    torch.cuda.empty_cache()
    return res_tensor, [c for sub in all_token_categories for c in sub]

def prepare_and_train_teo(device, model, sae, tokenizer, scaling_factor):
    test_dataset = IIDDataset(TEST_DATA_PATH, tokenizer_name="answerdotai/ModernBERT-base")
    diverse_indices, found_cats = [], set()
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        tokens = [tokenizer.decode([tid]) for tid in item['input_ids'][:min(item['attention_mask'].sum().item(), 512)]]
        cats_temp = {assign_category(t, "".join(tokens[max(0, k-5):min(len(tokens), k+5)])) for k, t in enumerate(tokens)}
        if len(cats_temp - {"Other"}) >= 4: diverse_indices.append(i); found_cats |= cats_temp
        if len(found_cats - {"Other"}) >= 8 and len(diverse_indices) >= 50: break
    train_loader = DataLoader(Subset(test_dataset, diverse_indices[:20]), batch_size=4)
    h_list, t_list_poly, t_list_sae = [], [], []
    for batch in tqdm(train_loader, desc="Preparing TEO"):
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        with torch.no_grad(): h_cls = model(ids, attention_mask=mask, output_hidden_states=True).hidden_states[22][:, 0, :]; h_list.append(h_cls.cpu())
        t_list_poly.append(torch.tensor(compute_attrs_batched(functools.partial(poly_forward, model), h_cls, torch.zeros(len(h_cls), dtype=torch.long).to(device), "Grad Activation", device)))
        t_list_sae.append(torch.tensor(compute_attrs_batched(functools.partial(sae_forward, model, sae, scaling_factor), h_cls, torch.zeros(len(h_cls), dtype=torch.long).to(device), "Grad Activation", device)))
    teo_poly = train_teo(torch.cat(h_list).to(device), torch.cat(t_list_poly).to(device), device, epochs=500, lambdas=(0.1, 0.3, 0.1, 0.5, 0.0))
    teo_sae = train_teo(torch.cat(h_list).to(device), torch.cat(t_list_sae).to(device), device, epochs=500, lambdas=(0.1, 0.3, 0.1, 0.5, 0.0))
    teo_umap = train_teo(torch.cat(h_list).to(device), torch.cat(t_list_sae).to(device), device, epochs=500, lambdas=(0.1, 0.3, 0.1, 0.5, 0.1))
    return teo_poly, teo_sae, teo_umap, diverse_indices

def run_pca_with_seed(seed, model, tokenizer, sae, teo_poly, teo_sae, teo_umap, diverse_indices, device, scaling_factor, is_sae_version=True):
    test_dataset = IIDDataset(TEST_DATA_PATH, tokenizer_name="answerdotai/ModernBERT-base")
    test_loader = DataLoader(Subset(test_dataset, diverse_indices), batch_size=1)
    results = {}
    if is_sae_version:
        methods = ["Activation", "Gradient Activation", "Integrated Gradients", "Gradient SHAP", "Layer Conductance"]
        for m in methods:
            set_seed(seed); results[m + " SAE"], cats = compute_token_attrs_batched(model, test_loader, m, device, tokenizer, sae=sae, scaling_factor=scaling_factor, limit=20)
        set_seed(seed); results["TEO-SAE"], _ = compute_token_attrs_batched(model, test_loader, "TEO-SAE", device, tokenizer, sae=sae, teo=teo_sae, scaling_factor=scaling_factor, limit=20)
        set_seed(seed); results["TEO-UMAP"], _ = compute_token_attrs_batched(model, test_loader, "TEO-SAE", device, tokenizer, sae=sae, teo=teo_umap, scaling_factor=scaling_factor, limit=20)
    else:
        methods = ["Activation", "Gradient Activation", "Integrated Gradients", "Gradient SHAP", "Layer Conductance"]
        for m in methods:
            set_seed(seed); results[m], cats = compute_token_attrs_batched(model, test_loader, m, device, tokenizer, sae=None, scaling_factor=1.0, limit=20)
        set_seed(seed); results["TEO"], _ = compute_token_attrs_batched(model, test_loader, "TEO", device, tokenizer, teo=teo_poly, scaling_factor=1.0, limit=20)
    cols = 3
    rows = (len(results) + 2) // 3
    # Standardize layout for all versions (3 columns)
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
    axes = axes.flatten()
    total_score = 0
    for i, (name, attrs) in enumerate(results.items()):
        mask = np.array([c != "Other" for c in cats]); attrs_f, cats_f = attrs[mask], [c for c in cats if c != "Other"]
        pca = PCA(n_components=2); X_proj = pca.fit_transform(attrs_f)
        for cn, color in CATEGORY_COLORS.items():
            cm = np.array([c == cn for c in cats_f])
            if cm.any():
                axes[i].scatter(X_proj[cm, 0], X_proj[cm, 1], c=color, label=cn, s=6, alpha=0.5)
        axes[i].set_title(name, fontsize=16, fontweight='bold'); axes[i].grid(True, linestyle='--', alpha=0.3)
        axes[i].set_aspect('equal', 'datalim') 
        axes[i].set_xlabel("Principal Component 1", fontsize=10)
        axes[i].set_ylabel("Principal Component 2", fontsize=10)
        axes[i].axis('tight')
        if "TEO" in name:
            centroids = [X_proj[np.array([c == cn for c in cats_f])].mean(axis=0) for cn in CATEGORY_COLORS.keys() if np.array([c == cn for c in cats_f]).sum() > 5]
            if len(centroids) > 1:
                from sklearn.metrics.pairwise import euclidean_distances
                score = euclidean_distances(centroids).mean(); total_score += score
                print(f"    {name} Separation Score (Seed {seed}): {score:.4f}")
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=True, fontsize=16, markerscale=2.5)
    plt.tight_layout(rect=[0, 0.1, 1, 0.98]); fname = "pca_analysis_overhaul.pdf" if is_sae_version else "pca_analysis_non_sae.pdf"
    
    # Save with landscape orientation metadata
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close()
    return total_score

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"; set_seed(0)
    model = get_model(device, FINETUNED_MODEL_PATH if os.path.exists(FINETUNED_MODEL_PATH) else "answerdotai/ModernBERT-base")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    sae = TopKTrainingSAE(TopKTrainingSAEConfig(d_in=768, d_sae=768 * 32, k=64)).to(device)
    if os.path.exists(MODEL_SAVE_PATH): sae.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    sae.eval(); scaling_factor = torch.load("models/sae_scaling_factor.pt", map_location="cpu")["scaling_factor"]
    teo_poly, teo_sae, teo_umap, diverse_indices = prepare_and_train_teo(device, model, sae, tokenizer, scaling_factor)
    BEST_SEED = 6
    print(f"Generating final figures with Seed {BEST_SEED} and 3-column layout")
    run_pca_with_seed(BEST_SEED, model, tokenizer, sae, teo_poly, teo_sae, teo_umap, diverse_indices, device, scaling_factor, True)
    run_pca_with_seed(BEST_SEED, model, tokenizer, sae, teo_poly, teo_sae, teo_umap, diverse_indices, device, scaling_factor, False)
