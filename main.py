import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf


def load_config(head_config_path: str, base_config_path: str = "configs/base.yaml") -> OmegaConf:
    """Loads and merges base.yaml and head-specific yaml with full interpolation resolution."""
    base_p = Path(base_config_path)
    head_p = Path(head_config_path)

    if not base_p.exists():
        raise FileNotFoundError(f"[CONFIG ERROR] Base config not found: {base_p.resolve()}")

    base_cfg = OmegaConf.load(str(base_p))
    head_cfg = OmegaConf.load(str(head_p)) if head_p.exists() else OmegaConf.create()

    # Merge configs: head-specific settings override base defaults
    cfg = OmegaConf.merge(base_cfg, head_cfg)
    OmegaConf.resolve(cfg)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="SOTIF Dual-Head Perception Supervisor Control Tower"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to execute")

    # --------------------------------------------------------------------------
    # Command 1: Feature Extraction
    # --------------------------------------------------------------------------
    extract_parser = subparsers.add_parser("extract", help="Extract ResNet-50 features to HDF5")
    extract_parser.add_argument("--data-dir", type=str, default="D:/SOTIF_supervision_workspace/data/raw/a2d2", help="Path to raw A2D2 images")
    extract_parser.add_argument("--output-h5", type=str, default="D:/SOTIF_supervision_workspace/data/features/train_features.h5", help="Target H5 path")
    extract_parser.add_argument("--batch-size", type=int, default=32)
    extract_parser.add_argument("--limit", type=int, default=None, help="Limit number of frames to extract (for dry run / sanity check)")

    # --------------------------------------------------------------------------
    # Command 2: Train Head A (Beta-VAE)
    # --------------------------------------------------------------------------
    train_a_parser = subparsers.add_parser("train-head-a", help="Train Head A Beta-VAE on extracted features")
    train_a_parser.add_argument("--config", type=str, default="configs/head_a.yaml", help="Path to Head A config file")
    train_a_parser.add_argument("--base-config", type=str, default="configs/base.yaml", help="Path to base system config file")

    # Optional CLI overrides (if omitted, values are pulled directly from YAML)
    train_a_parser.add_argument("--features-h5", type=str, default=None, help="Override features H5 path")
    train_a_parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    train_a_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    train_a_parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    train_a_parser.add_argument("--latent-dim", type=int, default=None, help="Override latent dimension")
    train_a_parser.add_argument("--beta", type=float, default=None, help="Override beta weighting factor")
    train_a_parser.add_argument("--track", action="store_true", default=None, help="Enable telemetry tracking (MLflow)")
    train_a_parser.add_argument("--tracking-uri", type=str, default=None, help="Override MLflow tracking URI")
    train_a_parser.add_argument("--experiment-name", type=str, default=None, help="Override MLflow experiment name")

    # --------------------------------------------------------------------------
    # Command 3: Compute Mahalanobis Parameters
    # --------------------------------------------------------------------------
    fit_mah_parser = subparsers.add_parser("fit-mahalanobis", help="Fit empirical mean and covariance on latent space")
    fit_mah_parser.add_argument("--features-h5", type=str, default="D:/SOTIF_supervision_workspace/data/features/train_features.h5")
    fit_mah_parser.add_argument("--checkpoint", type=str, default="checkpoints/head_a_best.pt")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- 1. FEATURE EXTRACTION ---
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

    # --- 2. TRAIN HEAD A (BETA-VAE) ---
    elif args.command == "train-head-a":
        # Load merged base + head_a configs via OmegaConf
        cfg = load_config(head_config_path=args.config, base_config_path=args.base_config)

        # Apply CLI overrides if explicitly passed by user
        features_h5 = args.features_h5 or cfg.storage.features.feature_h5
        epochs = args.epochs if args.epochs is not None else cfg.training.epochs
        batch_size = args.batch_size if args.batch_size is not None else cfg.training.batch_size
        lr = args.lr if args.lr is not None else cfg.training.lr
        latent_dim = args.latent_dim if args.latent_dim is not None else cfg.model.latent_dim
        beta = args.beta if args.beta is not None else cfg.training.beta

        # Tracking settings
        track_enabled = args.track if args.track is not None else cfg.storage.mlflow.get("tracking", False)
        tracking_uri = args.tracking_uri or cfg.storage.mlflow.tracking_uri
        exp_name = args.experiment_name or cfg.experiment.sub_experiment

        print("=" * 65)
        print("STAGE 02: HEAD A (BETA-VAE) TRAINING INITIALIZATION")
        print("=" * 65)
        print(f"[CONFIG] Features Container: {features_h5}")
        print(f"[CONFIG] Epochs:             {epochs}")
        print(f"[CONFIG] Batch Size:         {batch_size}")
        print(f"[CONFIG] Learning Rate:      {lr}")
        print(f"[CONFIG] Latent Dim (z):     {latent_dim}")
        print(f"[CONFIG] Beta Factor:        {beta}")
        print(f"[CONFIG] Tracking Active:    {track_enabled}")
        print(f"[CONFIG] MLflow URI:         {tracking_uri}")
        print(f"[CONFIG] Experiment Bucket:  {exp_name}")
        print("=" * 65)

        from src.common.tracking.factory import get_tracker
        from src.engine.trainer_vae import train_vae

        # Instantiate decoupled tracker (NoOp if track_enabled=False, MLflow if True)
        tracker = get_tracker(
            enabled=track_enabled,
            experiment_name=exp_name,
            tracking_uri=tracking_uri,
            run_name=f"{cfg.experiment.run_name}_z{latent_dim}_b{beta}"
        )

        with tracker:
            train_vae(
                features_h5=features_h5,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                latent_dim=latent_dim,
                beta=beta,
                val_split=cfg.training.val_split,
                checkpoint_dir=cfg.storage.models_dir,
                tracker=tracker
            )

    # --- 3. FIT MAHALANOBIS PARAMETERS ---
    elif args.command == "fit-mahalanobis":
        print("[INFO] Estimating latent centroid and precision matrix...")
        # from src.supervisor.mahalanobis import compute_and_save_density
        # compute_and_save_density(args.features_h5, args.checkpoint)


if __name__ == "__main__":
    main()