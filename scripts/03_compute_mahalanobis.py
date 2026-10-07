import sys
from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.head_a_vae import BetaVAE
from src.engine.trainer_vae import HDF5FeatureDataset


def build_sequence_aware_splits(h5_path: Path, val_ratio: float = 0.1):
  """Groups indices by the 14-digit drive timestamp prefix (e.g.

  '20181204191844').
  Allocates the first (1 - val_ratio) to train and the remaining val_ratio to
  val.
  """
  with h5py.File(h5_path, "r") as f:
    total_samples = f["features"].shape[0]
    has_filenames = "filenames" in f
    if has_filenames:
      raw_filenames = [
          name.decode("utf-8") if isinstance(name, bytes) else str(name)
          for name in f["filenames"][:]
      ]
    else:
      raw_filenames = None

  if (
      not has_filenames
      or raw_filenames is None
      or len(raw_filenames) != total_samples
  ):
    print(
        "[WARN] 'filenames' dataset not found in HDF5. Falling back to block"
        " chunking (90/10)."
    )
    chunk_size = max(100, total_samples // 10)
    train_idx, val_idx = [], []
    for start in range(0, total_samples, chunk_size):
      end = min(start + chunk_size, total_samples)
      block = list(range(start, end))
      split = int(len(block) * (1.0 - val_ratio))
      train_idx.extend(block[:split])
      val_idx.extend(block[split:])
    return train_idx, val_idx

  # Group by the true sequence ID (prefix before '_camera')
  groups = defaultdict(list)
  for idx, fname in enumerate(raw_filenames):
    base_name = Path(fname).name
    # Extract the timestamp prefix (e.g., '20181204191844')
    seq_id = base_name.split("_")[0]
    groups[seq_id].append(idx)

  train_idx = []
  val_idx = []

  print("\n--- Sequence Partition Summary ---")
  for seq_name, indices in groups.items():
    indices.sort()  # Maintain chronological order
    split_pt = int(len(indices) * (1.0 - val_ratio))
    train_idx.extend(indices[:split_pt])
    val_idx.extend(indices[split_pt:])
    print(
        f"  Drive '{seq_name}': {len(indices):6d} frames -> {split_pt:6d} Train"
        f" | {len(indices) - split_pt:6d} Val"
    )

  print(
      f"Total: {len(train_idx)} Train samples, {len(val_idx)} Validation"
      " samples\n"
  )
  return train_idx, val_idx


def main():
    base_cfg = OmegaConf.load("configs/base.yaml")
    head_cfg = OmegaConf.load("configs/head_a.yaml")
    cfg = OmegaConf.merge(base_cfg, head_cfg)
    OmegaConf.resolve(cfg)

    device = torch.device(cfg.compute.device if torch.cuda.is_available() else "cpu")

    features_h5 = Path(cfg.storage.features.feature_h5)
    checkpoint_path = Path(cfg.storage.models_dir) / "head_a_beta0.0001_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = Path(cfg.storage.models_dir) / "head_a_best.pt"

    out_params = Path(cfg.storage.root) / "mahalanobis_params.npz"

    print("=" * 70)
    print("STAGE 03: SEQUENCE-STRATIFIED MAHALANOBIS CALIBRATION")
    print("=" * 70)
    print(f"[INPUT] Model Checkpoint: {checkpoint_path}")
    print(f"[INPUT] Nominal Features: {features_h5}")

    # 1. Sequence-Stratified Partition
    dataset = HDF5FeatureDataset(str(features_h5))
    train_indices, val_indices = build_sequence_aware_splits(features_h5, val_ratio=cfg.training.val_split)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    # 2. Load Beta-VAE
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BetaVAE(in_features=cfg.model.input_dim, latent_dim=cfg.model.latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Collect Nominal Latent Vectors (Training Split)
    print("[INFO] Computing empirical distribution across training partition...")
    train_mu = []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            recon, mu, logvar, z = model(batch)
            train_mu.append(mu.cpu().numpy())

    train_mu = np.concatenate(train_mu, axis=0)  # [N_train, 32]

    # 4. Fit Empirical Centroid and Inverted Covariance Matrix
    mu_0 = np.mean(train_mu, axis=0)
    cov_0 = np.cov(train_mu, rowvar=False)

    # Tikhonov shrinkage regularization
    eps = 1e-4
    cov_reg = cov_0 + np.eye(cfg.model.latent_dim) * eps
    precision_0 = np.linalg.pinv(cov_reg)

    # 5. Calculate Validation Baseline Metrics to Derive Safety Thresholds
    print("[INFO] Deriving SOTIF safety thresholds on unseen sequence partitions...")
    val_dm = []
    val_mse = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, mu, logvar, z = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=1).cpu().numpy()
            mu_np = mu.cpu().numpy()

            delta = mu_np - mu_0
            dm = np.sqrt(np.sum(np.dot(delta, precision_0) * delta, axis=1))

            val_mse.extend(mse)
            val_dm.extend(dm)

    dataset.close()

    val_mse = np.array(val_mse)
    val_dm = np.array(val_dm)

    # Compute order statistics (95th and 99th percentiles)
    mse_th_95 = float(np.percentile(val_mse, 95))
    mse_th_99 = float(np.percentile(val_mse, 99))
    dm_th_95 = float(np.percentile(val_dm, 95))
    dm_th_99 = float(np.percentile(val_dm, 99))

    # 6. Save parameters to .npz
    np.savez(
        out_params,
        mu=mu_0,
        covariance=cov_0,
        precision=precision_0,
        mse_threshold_95=mse_th_95,
        mse_threshold_99=mse_th_99,
        dm_threshold_95=dm_th_95,
        dm_threshold_99=dm_th_99,
    )

    print("\n" + "=" * 70)
    print("CALIBRATION FINISHED")
    print("=" * 70)
    print(f"Nominal Centroid shape:       {mu_0.shape}")
    print(f"Precision Matrix shape:       {precision_0.shape}")
    print(f"MSE Validation Distribution:  mean={np.mean(val_mse):.6f}, std={np.std(val_mse):.6f}")
    print(f"DM Validation Distribution:   mean={np.mean(val_dm):.4f}, std={np.std(val_dm):.4f}")
    print(f"MSE 95% / 99% Thresholds:     {mse_th_95:.6f} / {mse_th_99:.6f}")
    print(f"D_M 95% / 99% Thresholds:     {dm_th_95:.4f} / {dm_th_99:.4f}")
    print(f"Persisted npz to:             {out_params}")
    print("=" * 70)


if __name__ == "__main__":
    main()