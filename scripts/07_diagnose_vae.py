import sys
from pathlib import Path
import argparse
import h5py
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from omegaconf import OmegaConf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.head_a_vae import BetaVAE
from utils.config import load_project_config


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Analytical KL divergence against standard Gaussian prior N(0, I):
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar), dim=-1)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


def get_a2d2_val_features(h5_path: str, val_ratio: float = 0.1) -> np.ndarray:
    """Extracts features from the sequence-stratified validation partition."""
    with h5py.File(h5_path, "r") as f:
        total = f["features"].shape[0]
        if "filenames" in f:
            names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in f["filenames"][:]]
            from collections import defaultdict
            groups = defaultdict(list)
            for idx, name in enumerate(names):
                seq_id = Path(name).name.split("_")[0]
                groups[seq_id].append(idx)
            val_indices = []
            for seq_id, indices in groups.items():
                indices.sort()
                split = int(len(indices) * (1.0 - val_ratio))
                val_indices.extend(indices[split:])
        else:
            # Fallback to contiguous 10% slice
            val_start = int(total * (1.0 - val_ratio))
            val_indices = list(range(val_start, total))

        val_features = f["features"][sorted(val_indices)]
    return val_features


def extract_video_features(video_path: str, resnet_backbone, global_pool, transform, device, crop_cfg):
    """Extracts 2048-d ResNet-50 GAP features for every frame in a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    features = []
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    top = int(orig_h * crop_cfg.get("top_ratio", 0.0)) if crop_cfg.get("enabled", False) else 0
    bottom = int(orig_h * crop_cfg.get("bottom_ratio", 1.0)) if crop_cfg.get("enabled", False) else orig_h

    print(f"[INFO] Extracting video features from: {Path(video_path).name} (Crop: {top} to {bottom})")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        eval_region = frame[top:bottom, :]
        rgb = cv2.cvtColor(eval_region, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            feat_map = resnet_backbone(tensor)
            feat = global_pool(feat_map).squeeze().cpu().numpy()
            features.append(feat)

    cap.release()
    return np.array(features)


def evaluate_vae(model, features_np, device, batch_size=256):
    """Computes L_rec, L_kl, mu_z, and sigma_z across all features."""
    dataset = torch.utils.data.TensorDataset(torch.tensor(features_np, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    l_rec_list = []
    l_kl_list = []
    mu_list = []
    sigma_list = []

    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar, z = model(batch)

            # L_rec: Mean Squared Error per sample (2048-d)
            rec_err = torch.mean((recon - batch) ** 2, dim=-1)
            # L_kl: Sum of KL divergence across latent dimensions
            kl_err = compute_kl_divergence(mu, logvar)
            # Posterior standard deviation: sigma = exp(0.5 * logvar)
            sigma = torch.exp(0.5 * logvar)

            l_rec_list.extend(rec_err.cpu().numpy())
            l_kl_list.extend(kl_err.cpu().numpy())
            mu_list.append(mu.cpu().numpy())
            sigma_list.append(sigma.cpu().numpy())

    return {
        "l_rec": np.array(l_rec_list),
        "l_kl": np.array(l_kl_list),
        "mu": np.concatenate(mu_list, axis=0),        # [N, 32]
        "sigma": np.concatenate(sigma_list, axis=0),  # [N, 32]
    }


def main():
    parser = argparse.ArgumentParser(description="VAE Diagnostic Suite")
    parser.add_argument("--video", type=str, required=True, help="Path to falsely flagged test video (.mp4)")
    parser.add_argument("--dataset-key", type=str, default="a2d2", help="Dataset crop key in configs/head_a.yaml")
    parser.add_argument("--out-dir", type=str, default="D:/SOTIF_supervision_workspace/evaluation_results/vae_diagnostics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Models and Config
    cfg = load_project_config(base_path="configs/base.yaml", head_path="configs/head_a.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ResNet-50 Feature Extractor
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet_backbone = torch.nn.Sequential(*list(resnet.children())[:-2]).to(device).eval()
    global_pool = torch.nn.AdaptiveAvgPool2d((1, 1)).to(device).eval()
    for p in resnet_backbone.parameters():
        p.requires_grad = False

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 512), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Beta-VAE Model
    checkpoint_path = Path(cfg.storage.models_dir) / f"head_a_beta{cfg.evaluation.target_beta}_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = Path(cfg.storage.models_dir) / "head_a_best.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae = BetaVAE(in_features=cfg.model.input_dim, latent_dim=cfg.model.latent_dim).to(device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()

    # 2. Extract Features from Both Sets
    h5_path = cfg.storage.features.feature_h5
    print(f"[INPUT] Loading A2D2 validation partition from: {h5_path}")
    a2d2_val_feats = get_a2d2_val_features(h5_path, val_ratio=cfg.training.val_split)

    crop_cfg = cfg.datasets.get(args.dataset_key, {}).get("roi_crop", {"enabled": False})
    test_feats = extract_video_features(args.video, resnet_backbone, global_pool, transform, device, crop_cfg)

    # 3. Compute the 4 Diagnostic Quantities
    print("[INFO] Evaluating VAE metrics on A2D2 validation...")
    a2d2_stats = evaluate_vae(vae, a2d2_val_feats, device)

    print("[INFO] Evaluating VAE metrics on test video...")
    test_stats = evaluate_vae(vae, test_feats, device)

    # 4. Print Formatted Comparison Table
    print("\n" + "=" * 78)
    print(f"{'Metric':<20} | {'A2D2 Validation':<25} | {'Test Video':<25}")
    print("=" * 78)

    rec_a2d2_str = f"{np.mean(a2d2_stats['l_rec']):.6f} ± {np.std(a2d2_stats['l_rec']):.6f}"
    rec_test_str = f"{np.mean(test_stats['l_rec']):.6f} ± {np.std(test_stats['l_rec']):.6f}"
    print(f"{'Recon error (L_rec)':<20} | {rec_a2d2_str:<25} | {rec_test_str:<25}")

    kl_a2d2_str = f"{np.mean(a2d2_stats['l_kl']):.4f} ± {np.std(a2d2_stats['l_kl']):.4f}"
    kl_test_str = f"{np.mean(test_stats['l_kl']):.4f} ± {np.std(test_stats['l_kl']):.4f}"
    print(f"{'KL Divergence (L_KL)':<20} | {kl_a2d2_str:<25} | {kl_test_str:<25}")

    mu_a2d2_str = f"{np.mean(a2d2_stats['mu']):.4f} ± {np.std(a2d2_stats['mu']):.4f}"
    mu_test_str = f"{np.mean(test_stats['mu']):.4f} ± {np.std(test_stats['mu']):.4f}"
    print(f"{'Latent μ_z':<20} | {mu_a2d2_str:<25} | {mu_test_str:<25}")

    sig_a2d2_str = f"{np.mean(a2d2_stats['sigma']):.4f} ± {np.std(a2d2_stats['sigma']):.4f}"
    sig_test_str = f"{np.mean(test_stats['sigma']):.4f} ± {np.std(test_stats['sigma']):.4f}"
    print(f"{'Latent σ_z':<20} | {sig_a2d2_str:<25} | {sig_test_str:<25}")
    print("=" * 78 + "\n")

    # 5. Plot 1: Reconstruction Error Distribution Comparison
    plt.figure(figsize=(9, 5))
    bins = np.linspace(
        0,
        max(np.percentile(a2d2_stats["l_rec"], 99.5), np.percentile(test_stats["l_rec"], 99.5)) * 1.2,
        60,
    )
    plt.hist(a2d2_stats["l_rec"], bins=bins, alpha=0.6, density=True, label="A2D2 Validation Set (ID)", color="teal")
    plt.hist(test_stats["l_rec"], bins=bins, alpha=0.6, density=True, label=f"Test Video: {Path(args.video).stem}", color="crimson")
    plt.axvline(np.mean(a2d2_stats["l_rec"]), color="teal", linestyle="--", linewidth=1.5, label="A2D2 Mean")
    plt.axvline(np.mean(test_stats["l_rec"]), color="crimson", linestyle="--", linewidth=1.5, label="Test Video Mean")
    plt.title("Reconstruction Error ($L_{rec}$) Distribution Comparison", fontsize=13, pad=12)
    plt.xlabel("Feature Reconstruction MSE", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_rec_path = out_dir / "recon_error_distribution_comparison.png"
    plt.savefig(plot_rec_path, dpi=300)
    plt.close()
    print(f"[SAVED] Reconstruction comparison plot: {plot_rec_path}")

    # 6. Plot 2: Latent Distribution Comparison (Across all 32 dimensions)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Latent mu distribution
    mu_bins = np.linspace(-3.5, 3.5, 60)
    axes[0].hist(a2d2_stats["mu"].flatten(), bins=mu_bins, alpha=0.6, density=True, label="A2D2 Validation μ", color="teal")
    axes[0].hist(test_stats["mu"].flatten(), bins=mu_bins, alpha=0.6, density=True, label="Test Video μ", color="crimson")
    axes[0].set_title("Latent Coordinate ($\mu_z$) Density Across 32 Dimensions", fontsize=12)
    axes[0].set_xlabel("Latent Coordinate Value", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Latent sigma distribution
    sig_bins = np.linspace(0.0, 2.0, 60)
    axes[1].hist(a2d2_stats["sigma"].flatten(), bins=sig_bins, alpha=0.6, density=True, label="A2D2 Validation σ", color="teal")
    axes[1].hist(test_stats["sigma"].flatten(), bins=sig_bins, alpha=0.6, density=True, label="Test Video σ", color="crimson")
    axes[1].set_title("Posterior Uncertainty ($\sigma_z = e^{0.5 \log \sigma^2}$) Density", fontsize=12)
    axes[1].set_xlabel("Posterior Standard Deviation", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_latent_path = out_dir / "latent_distribution_comparison.png"
    plt.savefig(plot_latent_path, dpi=300)
    plt.close()
    print(f"[SAVED] Latent comparison plot: {plot_latent_path}")


if __name__ == "__main__":
    main()