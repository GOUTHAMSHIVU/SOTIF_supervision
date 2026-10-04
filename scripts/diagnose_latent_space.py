import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.head_a_vae import BetaVAE
from src.engine.trainer_vae import HDF5FeatureDataset


def load_config(base_path: str = "configs/base.yaml", head_path: str = "configs/head_a.yaml") -> OmegaConf:
    base_cfg = OmegaConf.load(base_path)
    head_cfg = OmegaConf.load(head_path) if Path(head_path).exists() else OmegaConf.create()
    cfg = OmegaConf.merge(base_cfg, head_cfg)
    OmegaConf.resolve(cfg)
    return cfg


def get_validation_loader(features_h5: str, val_split: float, batch_size: int, seed: int):
    dataset = HDF5FeatureDataset(features_h5)
    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    _, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, val_loader, val_len


def main():
    parser = argparse.ArgumentParser(description="Head A Latent Diagnostic")
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Beta value to evaluate (e.g., 0.005, 1.0). If omitted, uses config.",
    )
    cli_args = parser.parse_args()

    # 1. Load config
    cfg = load_config()
    beta_val = cli_args.beta if cli_args.beta is not None else cfg.training.beta

    # 2. Determine target model and output path based on beta
    models_dir = Path(cfg.storage.models_dir)
    target_checkpoint = models_dir / f"head_a_beta{beta_val}_best.pt"
    if not target_checkpoint.exists():
        # Fallback to default best checkpoint if beta-tagged file doesn't exist
        target_checkpoint = models_dir / "head_a_best.pt"

    out_dir = Path(cfg.storage.root) / "diagnostics" / f"head_a_beta{beta_val}"
    out_dir.mkdir(parents=True, exist_ok=True)

    features_h5 = cfg.storage.features.feature_h5
    val_split = cfg.training.val_split
    seed = cfg.project.seed
    batch_size = cfg.diagnostics.get("batch_size", 256)
    var_threshold = cfg.diagnostics.get("variance_threshold", 0.01)
    latent_dim = cfg.model.latent_dim
    input_dim = cfg.model.input_dim

    device = torch.device(cfg.compute.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"HEAD A LATENT AUDIT: BETA = {beta_val}")
    print("=" * 70)
    print(f"[CONFIG] Checkpoint Target: {target_checkpoint}")
    print(f"[CONFIG] Output Directory:  {out_dir}")
    print(f"[CONFIG] Device:            {device}")
    print("=" * 70)

    # 3. Validation Loader
    dataset, val_loader, val_len = get_validation_loader(
        features_h5=features_h5,
        val_split=val_split,
        batch_size=batch_size,
        seed=seed,
    )

    # 4. Load Model
    checkpoint = torch.load(target_checkpoint, map_location=device)
    model = BetaVAE(in_features=input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_mu = []
    all_logvar = []
    all_mse = []

    # 5. Inference pass (calls forward directly)
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, mu, logvar, z = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=1)

            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_mse.append(mse.cpu().numpy())

    dataset.close()

    mu = np.concatenate(all_mu, axis=0)          # [N_val, 32]
    logvar = np.concatenate(all_logvar, axis=0)  # [N_val, 32]
    mse = np.concatenate(all_mse, axis=0)        # [N_val]
    sigma = np.exp(0.5 * logvar)

    # 6. Statistical Computation
    mu_mean = mu.mean(axis=0)
    mu_var = mu.var(axis=0)
    sigma_mean = sigma.mean(axis=0)
    sigma_var = sigma.var(axis=0)

    is_active = mu_var > var_threshold
    active_dims = int(np.sum(is_active))

    results_df = pd.DataFrame({
        "dimension": [f"z{i+1}" for i in range(latent_dim)],
        "mu_mean": mu_mean,
        "mu_variance": mu_var,
        "sigma_mean": sigma_mean,
        "sigma_variance": sigma_var,
        "status": ["Active" if a else "Collapsed" for a in is_active],
    })

    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.corrcoef(mu, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # 7. Persist Artifacts
    results_df.to_csv(out_dir / "latent_statistics.csv", index=False)
    np.save(out_dir / "latent_correlation.npy", corr_matrix)

    # 8. Report Breakdown
    pd.set_option("display.max_rows", 35)
    pd.set_option("display.width", 1000)
    print("\n--- LATENT SPACE PER-DIMENSION BREAKDOWN ---")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Mean Validation MSE:        {np.mean(mse):.6f}")
    print(f"Active Latent Dimensions:   {active_dims} / {latent_dim} (mu_var > {var_threshold})")
    print(f"Mean mu Variance across z:  {np.mean(mu_var):.6f}")
    print(f"Mean sigma across z:        {np.mean(sigma_mean):.6f}")
    print(f"Artifacts saved to:         {out_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()