import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.engine.trainer_vae import train_vae
from src.common.tracking.factory import get_tracker

def parse_args():
    parser = argparse.ArgumentParser(description="Train Head A Beta-VAE with decoupled tracking.")
    parser.add_argument("--features-h5", type=str, default="D:/SOTIF_supervision_workspace/features/train_features.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # Tracking flags
    parser.add_argument("--track", action="store_true", help="Enable experiment tracking")
    parser.add_argument("--tracking-uri", type=str, default="http://localhost:5000", help="MLflow server URI")
    parser.add_argument("--experiment-name", type=str, default="perception_ood", help="MLflow experiment bucket")
    parser.add_argument("--run-name", type=str, default=None, help="Descriptive run name")
    return parser.parse_args()

def main():
    args = parse_args()

    tracker = get_tracker(
        enabled=args.track,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        run_name=args.run_name or f"beta_vae_z{args.latent_dim}_b{args.beta}"
    )

    with tracker:
        train_vae(
            features_h5=args.features_h5,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            beta=args.beta,
            checkpoint_dir=args.checkpoint_dir,
            tracker=tracker
        )

if __name__ == "__main__":
    main()