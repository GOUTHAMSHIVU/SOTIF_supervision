import argparse
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="SOTIF Dual-Head Perception Supervisor Control Tower"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to execute")

    # Command 1: Feature Extraction
    extract_parser = subparsers.add_parser("extract", help="Extract ResNet-50 features to HDF5")
    extract_parser.add_argument("--data-dir", type=str, default="D:/SOTIF_supervision_workspace/data/raw/a2d2", help="Path to raw A2D2 images")
    extract_parser.add_argument("--output-h5", type=str, default="D:/SOTIF_supervision_workspace/features/train_features.h5", help="Target H5 path")
    extract_parser.add_argument("--batch-size", type=int, default=64)
    extract_parser.add_argument("--limit", type=int, default=None, help="Limit number of frames to extract (for dry run / sanity check)")

    # Command 2: Train Head A (VAE)
    train_a_parser = subparsers.add_parser("train-head-a", help="Train Head A Beta-VAE on extracted features")
    train_a_parser.add_argument("--features-h5", type=str, default="D:/SOTIF_supervision_workspace/features/train_features.h5")
    train_a_parser.add_argument("--epochs", type=int, default=50)
    train_a_parser.add_argument("--lr", type=float, default=1e-3)
    train_a_parser.add_argument("--latent-dim", type=int, default=32)

    # Command 3: Compute Mahalanobis Parameters
    fit_mah_parser = subparsers.add_parser("fit-mahalanobis", help="Fit empirical mean and covariance on latent space")
    fit_mah_parser.add_argument("--features-h5", type=str, default="D:/SOTIF_supervision_workspace/features/train_features.h5")
    fit_mah_parser.add_argument("--checkpoint", type=str, default="checkpoints/head_a_best.pt")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "extract":
        import importlib
        extract_module = importlib.import_module("scripts.01_extract_features")
        run_extraction = extract_module.run_extraction

        limit_str = f" (Dry run: first {args.limit} samples)" if args.limit else ""
        print(f"[INFO] Starting feature extraction pipeline...{limit_str}")
        run_extraction(
            data_dir=args.data_dir,
            output_h5=args.output_h5,
            batch_size=args.batch_size,
            limit=args.limit
        )

    elif args.command == "train-head-a":
        print(f"[INFO] Training Head A VAE for {args.epochs} epochs...")
        # from src.engine.trainer_vae import train_vae
        # train_vae(args.features_h5, epochs=args.epochs, lr=args.lr, latent_dim=args.latent_dim)

    elif args.command == "fit-mahalanobis":
        print("[INFO] Estimating latent centroid and precision matrix...")
        # from src.supervisor.mahalanobis import compute_and_save_density
        # compute_and_save_density(args.features_h5, args.checkpoint)

if __name__ == "__main__":
    main()