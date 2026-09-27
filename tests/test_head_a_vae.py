# tests/test_head_a_vae.py
import torch
import torch.nn.functional as F
from src.models.head_a_vae import SemanticOODSupervisor

def test_head_a_shapes(dummy_feature_vectors):
    latent_dim = 32
    model = SemanticOODSupervisor(in_features=2048, latent_dim=latent_dim)
    model.train()
    
    x_recon, mu, logvar, z = model(dummy_feature_vectors)
    
    assert x_recon.shape == (16, 2048), "Reconstruction vector must match input dimension"
    assert mu.shape == (16, latent_dim)
    assert logvar.shape == (16, latent_dim)
    assert z.shape == (16, latent_dim)

def test_head_a_gradient_backprop(dummy_feature_vectors):
    model = SemanticOODSupervisor(in_features=2048, latent_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    x_recon, mu, logvar, _ = model(dummy_feature_vectors)
    
    # Standard Beta-VAE loss computation
    recon_loss = F.mse_loss(x_recon, dummy_feature_vectors)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / dummy_feature_vectors.size(0)
    total_loss = recon_loss + 0.005 * kl_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    
    # Assert gradients exist and are not zero/NaN
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Zero gradient on {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient on {name}"
        
    optimizer.step()