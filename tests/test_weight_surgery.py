import torch
import pytest
from vision_baselines.model_3d import build_3d_model
from vision_baselines.weight_utils import adapt_swin_weights

def test_weight_slicing_logic():
    # 1. Create a "large" mock state dict (feature_size=48)
    large_model = build_3d_model(feature_size=48, pretrained=False)
    large_state = large_model.state_dict()
    
    # Fill with some identifiable values
    for k, v in large_state.items():
        if "weight" in k:
            torch.nn.init.constant_(v, 0.5)
        elif "bias" in k:
            torch.nn.init.constant_(v, 0.1)

    # 2. Create a "mini" model (feature_size=12)
    mini_model = build_3d_model(feature_size=12, pretrained=False)
    mini_state = mini_model.state_dict()
    
    # 3. Perform surgery
    adapted_state = adapt_swin_weights(large_state, mini_state)
    
    # 4. Load and verify
    mini_model.load_state_dict(adapted_state)
    
    # Verify some weights
    # Patch embed weight: [feature_size, 1, patch_size...]
    # stage 0 qkv weight: [3*feature_size, feature_size]
    
    # Check linear layer
    for name, param in mini_model.named_parameters():
        if "swin.swinViT.layers1.0.0.blocks.0.attn.qkv.weight" in name:
            assert param.shape == (3*12, 12)
            assert torch.allclose(param, torch.tensor(0.5))
        if "classifier.3.weight" in name: # Linear(enc_dim, 256)
             # enc_dim = 12 * 16 = 192 (for depths=(2,2,2,2))
             # v_src was 768 -> 192
             assert param.shape[1] == 12 * 16

def test_mini_model_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Small volumes for test speed
    model = build_3d_model(feature_size=12, depths=(2, 2, 2, 2), pretrained=False).to(device)
    dummy = torch.randn(2, 1, 64, 64, 64).to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    
    assert out["logits"].shape == (2,)
    assert not torch.isnan(out["logits"]).any()
    assert not torch.isinf(out["logits"]).any()
    print("Forward pass successful with mini configuration.")

if __name__ == "__main__":
    test_weight_slicing_logic()
    test_mini_model_forward()
