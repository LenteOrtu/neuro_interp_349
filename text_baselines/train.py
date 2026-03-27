import os
import torch
import wandb
import numpy as np
import random
import functools
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from itertools import cycle
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
from sae_lens.training.sae_trainer import SAETrainer
from captum.attr import InputXGradient, IntegratedGradients, GradientShap, FeatureAblation
import quantus
from teo import TEO, calculate_teo_loss

from typing import Generator, Optional, Dict, Any, Tuple
from data_loader import IIDDataset
from umer_secrets import WANDB_API_KEY

MODEL_SAVE_PATH = "models/sae_modernbert_baseline.pt"
FINETUNED_MODEL_PATH = "models/modernbert_adni_finetuned"
TRAIN_DATA_PATH = 'data/project_1_3_data/IID/ADNI_binary_training.csv'
TEST_DATA_PATH = 'data/project_1_3_data/IID/ADNI_binary_testing.csv'
BRAINLAT_TRAIN_DATA_PATH = 'data/project_1_3_data/OOD/binary_brainlat_training.csv'
BRAINLAT_TEST_DATA_PATH = 'data/project_1_3_data/OOD/binary_brainlat_testing.csv'
EPOCHS = 10 

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model(device: str, model_path: str = "answerdotai/ModernBERT-base", **kwargs: Any) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_path, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
    return model.to(device).eval()

def get_activations(model: torch.nn.Module, dataloader: DataLoader, layer: int = 22) -> tuple[torch.Tensor, float]:
    activations = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting activations"):
            mask = batch['attention_mask'].to(model.device)
            outputs = model.model(batch['input_ids'].to(model.device), 
                                attention_mask=mask, 
                                output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
            for i in range(hidden_states.shape[0]):
                real_tokens_len = mask[i].sum().item()
                activations.append(hidden_states[i, :real_tokens_len].cpu())
    all_acts = torch.cat(activations, dim=0)
    scaling_factor = all_acts.norm(dim=-1).mean().item()
    return all_acts / scaling_factor, scaling_factor

def get_pooled_latents(sae: TopKTrainingSAE, acts: torch.Tensor, device: str, d_sae: int, scaling_factor: float, pool: str = "cls") -> np.ndarray:
    latents = []
    acts = acts / scaling_factor
    with torch.no_grad():
        for i in range(0, len(acts), 4096):
            batch_acts = acts[i:i+4096].to(device)
            batch_latents = sae.encode(batch_acts).cpu().numpy()
            latents.append(batch_latents)
    full_latents = np.concatenate(latents, axis=0).reshape(-1, 512, d_sae)
    if pool == "cls":
        return full_latents[:, 0, :]
    return full_latents.mean(axis=1)

def cls_head(model: torch.nn.Module, h22_cls: torch.Tensor) -> torch.Tensor:
    return model.classifier(model.drop(model.head(h22_cls)))

def poly_forward(model: torch.nn.Module, h22_cls: torch.Tensor) -> torch.Tensor:
    return cls_head(model, h22_cls)

def sae_forward(model: torch.nn.Module, sae: torch.nn.Module, scaling_factor: float, h22_cls: torch.Tensor) -> torch.Tensor:
    h22_recon = sae(h22_cls / scaling_factor) * scaling_factor
    return cls_head(model, h22_recon)

def compute_attrs_batched(
    fwd_fn,
    h22: torch.Tensor,
    targets: torch.Tensor,
    method_name: str,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    results = []
    for i in range(0, len(h22), batch_size):
        bh = h22[i:i + batch_size].to(device)
        bt = targets[i:i + batch_size].to(device)
        if method_name == "Activations":
            results.append(bh.detach().cpu().numpy())
            continue
        if method_name == "Grad Activation":
            attr = InputXGradient(fwd_fn).attribute(bh, target=bt)
        elif method_name in ("Integrated Gradients", "Layer Conductance"):
            attr = IntegratedGradients(fwd_fn).attribute(bh, target=bt, n_steps=20)
        elif method_name == "Gradient SHAP":
            attr = GradientShap(fwd_fn).attribute(bh, baselines=torch.zeros_like(bh), target=bt)
        elif method_name == "Feature Ablation":
            attr = FeatureAblation(fwd_fn).attribute(bh, target=bt)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        results.append(attr.detach().cpu().numpy())
    return np.concatenate(results)

class QuantusModel(torch.nn.Module):
    def __init__(self, fwd_fn, device: str) -> None:
        super().__init__()
        self.fwd_fn = fwd_fn
        self.device = device
    def forward(self, inputs_np: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            t = torch.tensor(inputs_np, dtype=torch.float32).to(self.device)
            return self.fwd_fn(t).cpu()

def quantus_explain_fn(
    fwd_fn,
    method_name: str,
    _device: str,
    model,
    inputs: np.ndarray,
    targets: np.ndarray,
    **kwargs,
) -> np.ndarray:
    h = torch.tensor(inputs, dtype=torch.float32)
    t = torch.tensor(targets, dtype=torch.long)
    return compute_attrs_batched(fwd_fn, h, t, method_name, _device)

def train_teo(
    activations: torch.Tensor, 
    consensus_attributions: torch.Tensor, 
    device: str, 
    epochs: int = 150, 
    lr: float = 2e-4,
    lambdas: Optional[Tuple[float, float, float, float, float]] = None
) -> TEO:
    d_model = activations.shape[-1]
    model = TEO(d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        phi_hat = model(activations.to(device))
        if lambdas is not None:
            loss = calculate_teo_loss(phi_hat, consensus_attributions.to(device), activations.to(device), model, lambdas=lambdas)
        else:
            loss = calculate_teo_loss(phi_hat, consensus_attributions.to(device), activations.to(device), model)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Training [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")
    return model

def compute_classification_metrics(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    y_pred = (y_probs > 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_probs),
        "AP": average_precision_score(y_true, y_probs),
    }

def evaluate_sae(run_classification: bool = True, run_attribution: bool = False) -> None:
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = FINETUNED_MODEL_PATH if os.path.exists(FINETUNED_MODEL_PATH) else "answerdotai/ModernBERT-base"
    model = get_model(device, model_path)
    tokenizer_name = "answerdotai/ModernBERT-base"
    train_dataset = IIDDataset(TRAIN_DATA_PATH, tokenizer_name=tokenizer_name)
    test_dataset = IIDDataset(TEST_DATA_PATH, tokenizer_name=tokenizer_name)
    indices = np.random.permutation(len(train_dataset))
    train_idx = indices[:int(0.85 * len(train_dataset))]

    sae_cfg = TopKTrainingSAEConfig(d_in=768, d_sae=768 * 32, k=64)
    sae = TopKTrainingSAE(sae_cfg).to(device)
    sae.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    sae.eval()

    scaling_data = torch.load("models/sae_scaling_factor.pt", map_location="cpu")
    scaling_factor = scaling_data["scaling_factor"]

    print("Extracting training CLS latents for TEO and Probe...")
    train_h22_list, train_target_list, train_label_list = [], [], []
    with torch.no_grad():
        # Use 500 for the probe, but we'll use 200 for TEO
        for batch in tqdm(DataLoader(Subset(train_dataset, train_idx[:500]), batch_size=8), desc="Train extraction"):
            out = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), output_hidden_states=True)
            train_h22_list.append(out.hidden_states[22][:, 0, :].cpu())
            train_target_list.append(out.logits.argmax(dim=-1).cpu())
            train_label_list.append(batch['labels'].cpu())
    train_h22 = torch.cat(train_h22_list)
    train_targets = torch.cat(train_target_list)
    train_labels = torch.cat(train_label_list).numpy()

    print("Extracting test CLS latents...")
    test_h22_list, test_target_list, test_label_list, test_logits_list = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(DataLoader(test_dataset, batch_size=8), desc="Test extraction"):
            out = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), output_hidden_states=True)
            test_h22_list.append(out.hidden_states[22][:, 0, :].cpu())
            test_target_list.append(out.logits.argmax(dim=-1).cpu())
            test_label_list.append(batch['labels'].cpu())
            test_logits_list.append(out.logits.cpu())
    test_h22 = torch.cat(test_h22_list)
    test_targets = torch.cat(test_target_list)
    test_labels = torch.cat(test_label_list).numpy()
    test_logits_orig = torch.cat(test_logits_list)

    poly_fwd = functools.partial(poly_forward, model)
    sae_fwd = functools.partial(sae_forward, model, sae, scaling_factor)

    if run_classification:
        print("\nComputing classification metrics...")
        probs_orig = torch.softmax(test_logits_orig, dim=-1)[:, 1].numpy()
        metrics_orig = compute_classification_metrics(test_labels, probs_orig)

        # Logistic Regression Probe on SAE latents
        print("Training Logistic Regression Probe on SAE latents...")
        train_sae_latents = []
        with torch.no_grad():
            for i in range(0, len(train_h22), 32):
                bh = train_h22[i:i+32].to(device)
                train_sae_latents.append(sae.encode(bh / scaling_factor).cpu().numpy())
        train_sae_latents = np.concatenate(train_sae_latents)

        test_sae_latents = []
        with torch.no_grad():
            for i in range(0, len(test_h22), 32):
                bh = test_h22[i:i+32].to(device)
                test_sae_latents.append(sae.encode(bh / scaling_factor).cpu().numpy())
        test_sae_latents = np.concatenate(test_sae_latents)

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
        clf.fit(train_sae_latents, train_labels)
        probe_probs = clf.predict_proba(test_sae_latents)[:, 1]
        metrics_probe = compute_classification_metrics(test_labels, probe_probs)

        print(f" {'Metric':<15} | {'Model':>10} | {'SAE Probe':>10}")
        print("-" * 42)
        for m in ["Accuracy", "F1", "AUC", "AP"]:
            print(f"{m:<15} | {metrics_orig[m]:>10.4f} | {metrics_probe[m]:>10.4f}")

    if not run_attribution:
        return

    SUBSET_LIMIT = 50
    METHOD_NAMES = ["Grad Activation", "Integrated Gradients", "Layer Conductance", "Gradient SHAP", "Feature Ablation", "Activations"]
    
    print("\nComputing consensus attributions for TEO-Poly...")
    train_attrs_poly = []
    for method_name in METHOD_NAMES:
        a = compute_attrs_batched(poly_fwd, train_h22, train_targets, method_name, device)
        train_attrs_poly.append(a)
    consensus_phi_poly = torch.tensor(np.mean(train_attrs_poly, axis=0), dtype=torch.float32)

    print("Computing consensus attributions for TEO-SAE...")
    train_attrs_sae = []
    for method_name in METHOD_NAMES:
        a = compute_attrs_batched(sae_fwd, train_h22, train_targets, method_name, device)
        train_attrs_sae.append(a)
    consensus_phi_sae = torch.tensor(np.mean(train_attrs_sae, axis=0), dtype=torch.float32)

    print("\nTraining TEO-Poly...")
    teo_poly_model = train_teo(train_h22, consensus_phi_poly, device)
    teo_poly_model.eval()

    print("Training TEO-SAE...")
    teo_sae_model = train_teo(train_h22, consensus_phi_sae, device)
    teo_sae_model.eval()

    print("Training TEO-UMAP (SAE)...")
    # Section 3.6: UMAP constraints were most effective at the 4x batch size level.
    # We use a non-zero lambda5 for TEO-UMAP.
    teo_umap_model = train_teo(train_h22, consensus_phi_sae, device, lambdas=(0.1, 0.3, 0.1, 0.5, 0.1))
    teo_umap_model.eval()

    h22_sub = test_h22[:SUBSET_LIMIT]
    tgt_sub = test_targets[:SUBSET_LIMIT]
    x_np, y_np = h22_sub.numpy(), tgt_sub.numpy()

    all_attrs = {}
    SPACES = [("", poly_fwd), ("-SAE", sae_fwd)]
    for method_name in METHOD_NAMES:
        for suffix, fwd_fn in SPACES:
            label = method_name + suffix
            print(f"Computing: {label}")
            all_attrs[label] = compute_attrs_batched(fwd_fn, h22_sub, tgt_sub, method_name, device)

    with torch.no_grad():
        all_attrs["TEO"] = teo_poly_model(h22_sub.to(device)).cpu().numpy()
        all_attrs["TEO-SAE"] = teo_sae_model(h22_sub.to(device)).cpu().numpy()
        all_attrs["TEO-UMAP (SAE)"] = teo_umap_model(h22_sub.to(device)).cpu().numpy()

    print(f"\n{'Method':<30} | {'Sparseness':>10} | {'RIS':>8} | {'ROS':>8}")
    print("-" * 65)
    for label, a_batch in all_attrs.items():
        if "SAE" in label:
            fwd_fn = sae_fwd
            base_method = label.replace("-SAE", "").replace(" (SAE)", "")
        else:
            fwd_fn = poly_fwd
            base_method = label

        q_model = QuantusModel(fwd_fn, device).eval()
        
        if label == "TEO":
            expl_fn = lambda model, inputs, targets, **kwargs: teo_poly_model(torch.tensor(inputs, dtype=torch.float32).to(device)).detach().cpu().numpy()
        elif label == "TEO-SAE":
            expl_fn = lambda model, inputs, targets, **kwargs: teo_sae_model(torch.tensor(inputs, dtype=torch.float32).to(device)).detach().cpu().numpy()
        elif label == "TEO-UMAP (SAE)":
            expl_fn = lambda model, inputs, targets, **kwargs: teo_umap_model(torch.tensor(inputs, dtype=torch.float32).to(device)).detach().cpu().numpy()
        else:
            expl_fn = functools.partial(quantus_explain_fn, fwd_fn, base_method, device)
        
        sp = np.nanmean(quantus.Sparseness()(model=q_model, x_batch=x_np, y_batch=y_np, a_batch=a_batch, device="cpu"))
        ris = np.nanmean(quantus.RelativeInputStability(nr_samples=5, perturb_func_kwargs={"upper_bound": 0.1})(model=q_model, x_batch=x_np, y_batch=y_np, a_batch=a_batch, explain_func=expl_fn, device="cpu"))
        ros = np.nanmean(quantus.RelativeOutputStability(nr_samples=5, perturb_func_kwargs={"upper_bound": 0.1})(model=q_model, x_batch=x_np, y_batch=y_np, a_batch=a_batch, explain_func=expl_fn, device="cpu"))
        print(f"{label:<30} | {sp:>10.4f} | {ris:>8.4f} | {ros:>8.4f}")

def evaluate_sae_brainlat(run_classification: bool = True, run_attribution: bool = False) -> None:
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = FINETUNED_MODEL_PATH if os.path.exists(FINETUNED_MODEL_PATH) else "answerdotai/ModernBERT-base"
    model = get_model(device, model_path)
    tokenizer_name = "answerdotai/ModernBERT-base"
    
    # Brainlat OOD data
    test_dataset = IIDDataset(BRAINLAT_TEST_DATA_PATH, tokenizer_name=tokenizer_name)
    train_dataset = IIDDataset(BRAINLAT_TRAIN_DATA_PATH, tokenizer_name=tokenizer_name)

    sae_cfg = TopKTrainingSAEConfig(d_in=768, d_sae=768 * 32, k=64)
    sae = TopKTrainingSAE(sae_cfg).to(device)
    sae.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    sae.eval()

    scaling_data = torch.load("models/sae_scaling_factor.pt", map_location="cpu")
    scaling_factor = scaling_data["scaling_factor"]

    print("\n--- Evaluating Brainlat OOD ---")
    
    # 1. Classification Performance
    print("Extracting Brainlat latents for classification...")
    test_h22_list, test_target_list, y_true_list, logits_list = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(DataLoader(test_dataset, batch_size=8), desc="Brainlat extraction"):
            out = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), output_hidden_states=True)
            test_h22_list.append(out.hidden_states[22][:, 0, :].cpu())
            test_target_list.append(out.logits.argmax(dim=-1).cpu())
            y_true_list.append(batch['labels'].cpu())
            logits_list.append(out.logits.cpu())
    
    test_h22 = torch.cat(test_h22_list)
    test_targets = torch.cat(test_target_list)
    test_labels = torch.cat(y_true_list).numpy()
    all_logits = torch.cat(logits_list)

    poly_fwd = functools.partial(poly_forward, model)
    sae_fwd = functools.partial(sae_forward, model, sae, scaling_factor)

    if run_classification:
        print("\nComputing classification metrics...")
        # Base Model
        probs_model = torch.softmax(all_logits, dim=-1)[:, 1].numpy()
        metrics_model = compute_classification_metrics(test_labels, probs_model)

        # 2. Logistic Regression Probe on SAE latents (Brainlat)
        print("Extracting Brainlat training CLS latents for Probe...")
        train_h22_list, train_label_list = [], []
        with torch.no_grad():
            for batch in tqdm(DataLoader(Subset(train_dataset, range(min(len(train_dataset), 500))), batch_size=8), desc="Brainlat Train extraction"):
                out = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), output_hidden_states=True)
                train_h22_list.append(out.hidden_states[22][:, 0, :].cpu())
                train_label_list.append(batch['labels'].cpu())
        train_h22_LR = torch.cat(train_h22_list)
        train_labels_LR = torch.cat(train_label_list).numpy()

        print("Training Logistic Regression Probe on SAE latents (Brainlat)...")
        train_sae_latents = []
        with torch.no_grad():
            for i in range(0, len(train_h22_LR), 32):
                bh = train_h22_LR[i:i+32].to(device)
                train_sae_latents.append(sae.encode(bh / scaling_factor).cpu().numpy())
        train_sae_latents = np.concatenate(train_sae_latents)

        test_sae_latents = []
        with torch.no_grad():
            for i in range(0, len(test_h22), 32):
                bh = test_h22[i:i+32].to(device)
                test_sae_latents.append(sae.encode(bh / scaling_factor).cpu().numpy())
        test_sae_latents = np.concatenate(test_sae_latents)

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
        clf.fit(train_sae_latents, train_labels_LR)
        probe_probs = clf.predict_proba(test_sae_latents)[:, 1]
        metrics_probe = compute_classification_metrics(test_labels, probe_probs)

        print(f"{'Metric':<15} | {'Model':>10} | {'SAE Probe':>10}")
        print("-" * 42)
        for m in ["Accuracy", "F1", "AUC", "AP"]:
            print(f"{m:<15} | {metrics_model[m]:>10.4f} | {metrics_probe[m]:>10.4f}")

    if not run_attribution:
        return

    # 3. Attribution Results
    print("\nExtracting Brainlat training CLS latents for TEO...")
    train_h22_teo_list, train_target_teo_list = [], []
    with torch.no_grad():
        for batch in tqdm(DataLoader(Subset(train_dataset, range(min(len(train_dataset), 200))), batch_size=8), desc="Brainlat Train extraction TEO"):
            out = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), output_hidden_states=True)
            train_h22_teo_list.append(out.hidden_states[22][:, 0, :].cpu())
            train_target_teo_list.append(out.logits.argmax(dim=-1).cpu())
    train_h22_teo = torch.cat(train_h22_teo_list)
    train_targets_teo = torch.cat(train_target_teo_list)

    METHOD_NAMES = ["Grad Activation", "Integrated Gradients", "Layer Conductance", "Gradient SHAP", "Feature Ablation", "Activations"]
    
    print("\nComputing consensus attributions for TEO-Poly (Brainlat)...")
    train_attrs_poly = []
    for method_name in METHOD_NAMES:
        a = compute_attrs_batched(poly_fwd, train_h22_teo, train_targets_teo, method_name, device)
        train_attrs_poly.append(a)
    consensus_phi_poly = torch.tensor(np.mean(train_attrs_poly, axis=0), dtype=torch.float32)

    print("Computing consensus attributions for TEO-SAE (Brainlat)...")
    train_attrs_sae = []
    for method_name in METHOD_NAMES:
        a = compute_attrs_batched(sae_fwd, train_h22_teo, train_targets_teo, method_name, device)
        train_attrs_sae.append(a)
    consensus_phi_sae = torch.tensor(np.mean(train_attrs_sae, axis=0), dtype=torch.float32)

    print("\nTraining TEO-Poly (Brainlat)...")
    teo_poly_model = train_teo(train_h22_teo, consensus_phi_poly, device)
    teo_poly_model.eval()

    print("Training TEO-SAE (Brainlat)...")
    teo_sae_model = train_teo(train_h22_teo, consensus_phi_sae, device)
    teo_sae_model.eval()

    print("Training TEO-UMAP (SAE) (Brainlat)...")
    teo_umap_model = train_teo(train_h22_teo, consensus_phi_sae, device, lambdas=(0.1, 0.3, 0.1, 0.5, 0.1))
    teo_umap_model.eval()

    SUBSET_LIMIT = 50
    h22_sub = test_h22[:SUBSET_LIMIT]
    tgt_sub = test_targets[:SUBSET_LIMIT]
    x_np, y_np = h22_sub.numpy(), tgt_sub.numpy()

    all_attrs = {}
    SPACES = [("", poly_fwd), ("-SAE", sae_fwd)]
    for method_name in METHOD_NAMES:
        for suffix, fwd_fn in SPACES:
            label = method_name + suffix
            print(f"Computing: {label}")
            all_attrs[label] = compute_attrs_batched(fwd_fn, h22_sub, tgt_sub, method_name, device)

    with torch.no_grad():
        all_attrs["TEO"] = teo_poly_model(h22_sub.to(device)).cpu().numpy()
        all_attrs["TEO-SAE"] = teo_sae_model(h22_sub.to(device)).cpu().numpy()
        all_attrs["TEO-UMAP (SAE)"] = teo_umap_model(h22_sub.to(device)).cpu().numpy()

    print(f"\n{'Method':<30} | {'Sparseness':>10} | {'RIS':>8} | {'ROS':>8}")
    print("-" * 65)
    for label, a_batch in all_attrs.items():
        if "SAE" in label:
            fwd_fn = sae_fwd
            base_method = label.replace("-SAE", "").replace(" (SAE)", "")
        else:
            fwd_fn = poly_fwd
            base_method = label

        q_model = QuantusModel(fwd_fn, device).eval()
        
        if label == "TEO":
            expl_fn = lambda model, inputs, targets, **kwargs: teo_poly_model(torch.tensor(inputs, dtype=torch.float32).to(device)).detach().cpu().numpy()
        elif label == "TEO-SAE":
            expl_fn = lambda model, inputs, targets, **kwargs: teo_sae_model(torch.tensor(inputs, dtype=torch.float32).to(device)).detach().cpu().numpy()
        elif label == "TEO-UMAP (SAE)":
            expl_fn = lambda model, inputs, targets, **kwargs: teo_umap_model(torch.tensor(inputs, dtype=torch.float32).to(device)).detach().cpu().numpy()
        else:
            expl_fn = functools.partial(quantus_explain_fn, fwd_fn, base_method, device)
        
        sp = np.nanmean(quantus.Sparseness()(model=q_model, x_batch=x_np, y_batch=y_np, a_batch=a_batch, device="cpu"))
        ris = np.nanmean(quantus.RelativeInputStability(nr_samples=5, perturb_func_kwargs={"upper_bound": 0.1})(model=q_model, x_batch=x_np, y_batch=y_np, a_batch=a_batch, explain_func=expl_fn, device="cpu"))
        ros = np.nanmean(quantus.RelativeOutputStability(nr_samples=5, perturb_func_kwargs={"upper_bound": 0.1})(model=q_model, x_batch=x_np, y_batch=y_np, a_batch=a_batch, explain_func=expl_fn, device="cpu"))
        print(f"{label:<30} | {sp:>10.4f} | {ris:>8.4f} | {ros:>8.4f}")

if __name__ == "__main__":
    print("Evaluating ADNI...")
    evaluate_sae(run_classification=True, run_attribution=False)
    print("\n" + "="*80)
    print("Evaluating Brainlat...")
    evaluate_sae_brainlat(run_classification=True, run_attribution=False)
