import torch
import torch.nn as nn
from typing import Tuple

class TEO(nn.Module):
    """
    Transformer Explanation Optimizer (TEO) matching the paper's architecture.
    """
    def __init__(self, d_model: int = 768, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 2048):
        super().__init__()
        # Section 3.2: x-transformer autoencoder [27, 28]
        # We use a standard Transformer Encoder-Decoder architecture.
        # d_model is the dimensionality of the input features (e.g., 768 or SAE latents).
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_layers
        )
        # Final projection to the same attribution space as the input features
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TEO.
        Args:
            x (torch.Tensor): Activation tensor of shape [Batch, HiddenSize]
        Returns:
            torch.Tensor: Optimized attribution scores of shape [Batch, HiddenSize]
        """
        # Treat the input as a sequence of length 1 for the Transformer
        x_seq = x.unsqueeze(1)
        # Encoder processes the original input activations
        memory = self.encoder(x_seq)
        # Decoder generates the reconstructed explanation from the input and memory
        out = self.decoder(x_seq, memory)
        # Project to target explanation space (same size as input features)
        phi_hat = self.head(out).squeeze(1)
        return phi_hat

def calculate_teo_loss(
    phi_hat: torch.Tensor,
    phi_bar: torch.Tensor,
    x: torch.Tensor,
    model: TEO,
    lambdas: Tuple[float, float, float, float, float] = (0.1, 0.3, 0.1, 0.5, 0.1)
) -> torch.Tensor:
    """
    Total cost function for training the TEO model as defined in Equation 3 and Section 3.3 of the paper.
    
    L_total(phi^(k), phi_hat) = lambda1 * (1/M_RIS) + lambda2 * (1/M_ROS) + lambda3 * M_sparse + lambda4 * L_similarity + lambda5 * L_umap
    
    Args:
        phi_hat (torch.Tensor): Reconstructed attribution from TEO [Batch, HiddenSize]
        phi_bar (torch.Tensor): Consensus attribution target [Batch, HiddenSize]
        x (torch.Tensor): Input activations [Batch, HiddenSize]
        model (TEO): TEO model for generating stability proxies
        lambdas (tuple): Hyperparameters (l1, l2, l3, l4, l5)
    Returns:
        torch.Tensor: Total loss
    """
    l1, l2, l3, l4, l5 = lambdas
    eps = 1e-8
    
    # 1. Similarity Loss (Eq 3) - L_similarity(phi_hat, phi_bar)
    loss_similarity = nn.functional.mse_loss(phi_hat, phi_bar)
    
    # 2. Sparseness Penalty (Eq 3) - M_sparse
    # We use L1 norm as a differentiable proxy for concentrated attributions.
    loss_sparse = torch.mean(torch.abs(phi_hat))
    
    # 3. Stability Proxies (RIS and ROS)
    # The paper uses 1/M_RIS and 1/M_ROS in the cost function.
    noise = torch.randn_like(x) * 0.01
    phi_hat_perturbed = model(x + noise)
    
    # Instability is the squared difference
    instability_ris = torch.mean((phi_hat - phi_hat_perturbed)**2)
    m_ris_proxy = 1.0 / (instability_ris + eps)
    
    # Proxy for ROS (Relative Output Stability)
    instability_ros = instability_ris 
    m_ros_proxy = 1.0 / (instability_ros + eps)

    # 4. UMAP Linear Constraint (Section 3.3) - lambda5
    # We implement a differentiable proxy using PCA for the "geometry-aware constraint".
    # We apply feature-wise projection: each feature is a point in sample space (BatchSize).
    # phi_hat has shape [Batch, HiddenSize]. We treat it as [HiddenSize, Batch].
    try:
        # Normalize features (columns) min-max to [0, 1] as per Section 3.3
        phi_min = phi_hat.min(dim=0, keepdim=True)[0]
        phi_max = phi_hat.max(dim=0, keepdim=True)[0]
        phi_norm = (phi_hat - phi_min) / (phi_max - phi_min + eps)
        
        # PCA on features: [HiddenSize, BatchSize]
        # We want to embed HiddenSize features into 2D space.
        A = phi_norm.t() 
        # torch.pca_lowrank returns U, S, V such that A approx U S V.t()
        # The projected coordinates are U * S.
        U, S, V = torch.pca_lowrank(A, q=2)
        proj = torch.matmul(U, torch.diag(S)) # [HiddenSize, 2]
        
        # Linear constraint: u_i1 = u_i2 -> (u_i1 - u_i2)^2
        loss_umap = torch.mean((proj[:, 0] - proj[:, 1])**2)
    except Exception:
        # Fallback if PCA fails (e.g. batch size too small)
        loss_umap = torch.tensor(0.0, device=phi_hat.device)
    
    # Equation 3 + Section 3.3 implementation.
    total_loss = (l1 * (1.0 / m_ris_proxy)) + (l2 * (1.0 / m_ros_proxy)) + (l3 * loss_sparse) + (l4 * loss_similarity) + (l5 * loss_umap)
    
    return total_loss
