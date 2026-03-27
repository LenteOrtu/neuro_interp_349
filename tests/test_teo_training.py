import torch
import numpy as np
import pytest
import sys
import os

# Adjust path to import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_baselines.teo import TEO, calculate_teo_loss

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def test_teo_forward_shape():
    """Verify that TEO correctly handles [Batch, HiddenSize] input."""
    batch_size = 4
    d_model = 768
    model = TEO(d_model=d_model)
    x = torch.randn(batch_size, d_model)
    phi_hat = model(x)
    assert phi_hat.shape == (batch_size, d_model)

def test_teo_loss_decreases(device):
    """Verify that the loss decreases during training on synthetic data."""
    torch.manual_seed(0)
    batch_size = 8
    d_model = 768
    model = TEO(d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Synthetic activations and target consensus attributions (phi_bar)
    x = torch.randn(batch_size, d_model).to(device)
    phi_bar = torch.randn(batch_size, d_model).to(device)
    
    model.train()
    initial_loss = calculate_teo_loss(model(x), phi_bar, x, model).item()
    
    for _ in range(20):
        optimizer.zero_grad()
        phi_hat = model(x)
        loss = calculate_teo_loss(phi_hat, phi_bar, x, model)
        loss.backward()
        optimizer.step()
        
    final_loss = calculate_teo_loss(model(x), phi_bar, x, model).item()
    assert final_loss < initial_loss

def test_teo_stability_proxy(device):
    """Check that stability loss term is computable."""
    torch.manual_seed(0)
    d_model = 128
    model = TEO(d_model=d_model).to(device)
    x = torch.randn(2, d_model).to(device)
    phi_bar = torch.zeros(2, d_model).to(device)
    
    # Run a forward pass and loss calculation
    phi_hat = model(x)
    loss = calculate_teo_loss(phi_hat, phi_bar, x, model, lambdas=(1.0, 1.0, 0.0, 0.0))
    
    assert loss >= 0
    assert not torch.isnan(loss)
