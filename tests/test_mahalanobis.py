# tests/test_mahalanobis.py
import torch

def test_mahalanobis_covariance_inversion():
    torch.manual_seed(42)
    # 500 samples in 32-d latent space
    nominal_latents = torch.randn(500, 32, dtype=torch.float32)
    
    mu = torch.mean(nominal_latents, dim=0)
    diff = nominal_latents - mu
    cov = torch.mm(diff.t(), diff) / (nominal_latents.size(0) - 1)
    
    # Add regularizer to guarantee positive-definiteness
    cov += 1e-4 * torch.eye(32)
    
    # Invert covariance matrix in FP32
    inv_cov = torch.linalg.inv(cov)
    
    assert inv_cov.shape == (32, 32)
    assert not torch.isinf(inv_cov).any(), "Covariance matrix inversion exploded to Inf"
    assert not torch.isnan(inv_cov).any(), "Covariance matrix inversion produced NaNs"

    # Test distance metric discrimination:
    # 1. An In-Distribution point (close to mean)
    id_point = mu + 0.1 * torch.randn(32)
    delta_id = (id_point - mu).unsqueeze(0)
    d_m_id = torch.sqrt(torch.mm(torch.mm(delta_id, inv_cov), delta_id.t()))

    # 2. An Out-of-Distribution point (extreme offset)
    ood_point = mu + 8.0 * torch.randn(32)
    delta_ood = (ood_point - mu).unsqueeze(0)
    d_m_ood = torch.sqrt(torch.mm(torch.mm(delta_ood, inv_cov), delta_ood.t()))

    assert d_m_ood > d_m_id, "OOD Mahalanobis distance must exceed Nominal distance"