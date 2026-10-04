from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Optional, Union

import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.models.head_a_vae import BetaVAE
from src.common.tracking.base import BaseTracker
from src.common.tracking.noop_tracker import NoOpTracker


class HDF5FeatureDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = str(Path(h5_path).resolve())
        with h5py.File(self.h5_path, "r") as f:
            self.num_samples = f["features"].shape[0]
            self.feature_dim = f["features"].shape[1]
        self._h5_file = None

    def _open_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._open_file()
        return torch.from_numpy(self._h5_file["features"][idx]).float()

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None


def loss_function(x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta * kld_loss, recon_loss, kld_loss


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "UNKNOWN"


def train_vae(
    features_h5: Optional[str] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    latent_dim: Optional[int] = None,
    beta: Optional[float] = None,
    val_split: Optional[float] = None,
    checkpoint_dir: Optional[str] = None,
    cfg: Optional[Union[Dict[str, Any], Any]] = None,
    tracker: Optional[BaseTracker] = None,
):
    """
    Trains Head A Beta-VAE on nominal feature vectors using pluggable tracking.
    Parameters can be passed directly via keyword arguments or via the unified `cfg` object.
    """
    if tracker is None:
        tracker = NoOpTracker()

    # --- 1. Parameter Resolution (cfg fallback) ---
    if cfg is not None:
        features_h5 = features_h5 or cfg.storage.features.feature_h5
        epochs = epochs if epochs is not None else cfg.training.epochs
        lr = lr if lr is not None else cfg.training.lr
        batch_size = batch_size if batch_size is not None else cfg.training.batch_size
        latent_dim = latent_dim if latent_dim is not None else cfg.model.latent_dim
        beta = beta if beta is not None else cfg.training.beta
        val_split = val_split if val_split is not None else cfg.training.get("val_split", 0.1)
        checkpoint_dir = checkpoint_dir or cfg.storage.get("models_dir", "models")
    else:
        # Default fallbacks if neither cfg nor individual args are provided
        features_h5 = features_h5 or "D:/SOTIF_supervision_workspace/features/train_features.h5"
        epochs = epochs if epochs is not None else 50
        lr = lr if lr is not None else 1e-3
        batch_size = batch_size if batch_size is not None else 128
        latent_dim = latent_dim if latent_dim is not None else 32
        beta = beta if beta is not None else 1.0
        val_split = val_split if val_split is not None else 0.1
        checkpoint_dir = checkpoint_dir or "models"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Dataset Setup ---
    dataset = HDF5FeatureDataset(features_h5)
    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    train_set, val_set = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # --- 3. Model, Optimizer, Scheduler ---
    model = BetaVAE(in_features=dataset.feature_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    checkpoint_p = Path(checkpoint_dir)
    checkpoint_p.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_p / f"head_a_beta{beta}_best.pt"

    # --- 4. Parameter Telemetry Logging ---
    tracker.log_params({
        "pipeline_stage": "head_a_ood_detection",
        "latent_dim": latent_dim,
        "beta": beta,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "val_split": val_split,
        "input_feature_dim": dataset.feature_dim,
        "total_samples": total_len,
        "features_source": str(Path(features_h5).resolve()),
        "git_commit": get_git_commit(),
        "device": str(device)
    })

    best_val_loss = float("inf")

    # --- 5. Training Loop ---
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.perf_counter()

        model.train()
        t_loss, t_mse, t_kld = 0.0, 0.0, 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar,z = model(batch)
            loss, mse, kld = loss_function(batch, recon, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * batch.size(0)
            t_mse += mse.item() * batch.size(0)
            t_kld += kld.item() * batch.size(0)

        t_loss /= train_len
        t_mse /= train_len
        t_kld /= train_len

        # Validation
        model.eval()
        v_loss, v_mse, v_kld = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar, z = model(batch)
                loss, mse, kld = loss_function(batch, recon, mu, logvar, beta=beta)
                v_loss += loss.item() * batch.size(0)
                v_mse += mse.item() * batch.size(0)
                v_kld += kld.item() * batch.size(0)

        v_loss /= val_len
        v_mse /= val_len
        v_kld /= val_len

        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_duration = time.perf_counter() - epoch_start_time

        # Telemetry metrics logged directly to tracker per epoch
        tracker.log_metrics({
            "epoch_duration_sec": epoch_duration,
            "train/loss": t_loss,
            "train/mse": t_mse,
            "train/kld": t_kld,
            "val/loss": v_loss,
            "val/mse": v_mse,
            "val/kld": v_kld,
            "lr": current_lr
        }, step=epoch)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({epoch_duration:.2f}s) | "
            f"Train: {t_loss:.4f} (MSE: {t_mse:.4f}, KLD: {t_kld:.4f}) | "
            f"Val: {v_loss:.4f}"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "latent_dim": latent_dim,
                "input_dim": dataset.feature_dim,
                "val_loss": v_loss,
                "beta": beta
            }, best_checkpoint_path)
            tracker.log_metrics({"val/best_loss": best_val_loss}, step=epoch)

    # Log best weights artifact to tracker
    tracker.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
    dataset.close()
    print(f"\n[ENGINE] Training completed. Best checkpoint: {best_checkpoint_path.resolve()}")