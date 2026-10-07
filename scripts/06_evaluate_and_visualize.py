import sys
from pathlib import Path
import re
import argparse
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.head_a_vae import BetaVAE
from utils.config import load_project_config


def natural_sort_key(s):
    """Sorts strings containing numbers in numerical order (1, 2, ..., 10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


class VisualSOTIFEvaluator:
    def __init__(self, config_path: str = "configs/head_a.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load Configurations
        self.cfg = load_project_config(base_path="configs/base.yaml", head_path=config_path)
        target_beta = self.cfg.evaluation.get("target_beta", 0.0001)

        # 2. ResNet-50 Feature Extractor
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.conv_backbone = torch.nn.Sequential(*list(resnet.children())[:-2]).to(self.device).eval()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1)).to(self.device).eval()

        # Input resolution matching nominal feature extraction (256x512)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 512), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 3. Load Trained Head A (Beta-VAE)
        models_dir = Path(self.cfg.storage.models_dir)
        checkpoint_path = models_dir / f"head_a_beta{target_beta}_best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = models_dir / "head_a_best.pt"

        print(f"[SUPERVISOR] Loading Head A weights: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.vae = BetaVAE(in_features=self.cfg.model.input_dim, latent_dim=self.cfg.model.latent_dim).to(self.device)
        self.vae.load_state_dict(checkpoint["model_state_dict"])
        self.vae.eval()

        # 4. Load Calibrated Mahalanobis Parameters (Stage 03)
        params_path = Path(self.cfg.evaluation.calibrated_params)
        if not params_path.exists():
            raise FileNotFoundError(f"Calibrated parameters not found at {params_path}. Run Stage 03 first.")

        params = np.load(params_path)
        self.mu_nom_np = params["mu"]                 # [32]
        self.precision_np = params["precision"]       # [32, 32]
        
        self.mse_th = float(params.get("mse_threshold_95", 0.007843))
        self.dm_th = float(params.get("dm_threshold_95", 5.9761))
        print(f"[SUPERVISOR] Calibrated Thresholds: MSE_th={self.mse_th:.6f}, D_M_th={self.dm_th:.4f}")

        # 5. Temporal Filtering Settings
        self.window_size = self.cfg.evaluation.temporal.get("window_size", 5)
        self.mse_buffer = []
        self.dm_buffer = []

    def reset_buffer(self):
        """Clears sliding window across distinct test scenes."""
        self.mse_buffer.clear()
        self.dm_buffer.clear()

    def inspect_frame(self, frame_bgr, crop_cfg):
        orig_h, orig_w, _ = frame_bgr.shape
        annotated = frame_bgr.copy()

        # 1. Camera ROI Crop (Only strips ego-vehicle hood / mounting timestamps)
        top_crop = 0
        bottom_crop = orig_h
        if crop_cfg.get("enabled", False):
            top_ratio = crop_cfg.get("top_ratio", 0.0)
            bottom_ratio = crop_cfg.get("bottom_ratio", 1.0)
            top_crop = int(orig_h * top_ratio)
            bottom_crop = int(orig_h * bottom_ratio)
            eval_region = frame_bgr[top_crop:bottom_crop, :].copy()
        else:
            eval_region = frame_bgr.copy()

        eval_h, eval_w, _ = eval_region.shape

        # 2. ResNet-50 Forward Pass with Gradient Retention for Grad-CAM
        rgb = cv2.cvtColor(eval_region, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        feat_map = self.conv_backbone(tensor)           # [1, 2048, H, W]
        feat_map.retain_grad()

        x_gap = self.global_pool(feat_map).squeeze(-1).squeeze(-1)  # [1, 2048]

        # 3. Head A (Beta-VAE) Reconstruction & Latent Scoring
        recon, mu, _, _ = self.vae(x_gap)
        recon_loss = F.mse_loss(recon, x_gap)
        raw_mse = float(recon_loss.item())

        mu_np = mu.squeeze(0).detach().cpu().numpy()
        delta = mu_np - self.mu_nom_np
        raw_dm = float(np.sqrt(np.dot(np.dot(delta, self.precision_np), delta)))

        # 4. Temporal Smoothing via Moving Average Buffer
        self.mse_buffer.append(raw_mse)
        self.dm_buffer.append(raw_dm)
        if len(self.mse_buffer) > self.window_size:
            self.mse_buffer.pop(0)
            self.dm_buffer.pop(0)

        filt_mse = float(np.mean(self.mse_buffer))
        filt_dm = float(np.mean(self.dm_buffer))

        # 5. SOTIF Gate: Check for Out-of-Distribution Hazard
        is_hazard = (filt_mse > self.mse_th) or (filt_dm > self.dm_th)

        # 6. Grad-CAM Spatial Attribution Heatmap
        recon_loss.backward()
        grads = feat_map.grad  # [1, 2048, H, W]

        if grads is not None:
            # Global average pool the gradients along spatial axes
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * feat_map, dim=1).squeeze(0)
            cam = F.relu(cam)
            cam_np = cam.detach().cpu().numpy()

            if np.max(cam_np) > 0:
                cam_np = cam_np / np.max(cam_np)
            cam_resized = cv2.resize(cam_np, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

            # Generate Jet colormap
            heatmap_raw = np.uint8(255 * cam_resized)
            heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)

            # Apply overlay only when hazard is triggered or threshold is approached
            if is_hazard or filt_mse > (self.mse_th * 0.85):
                # Alpha blend heatmap onto evaluated image region
                alpha = 0.40
                eval_overlay = cv2.addWeighted(eval_region, 1.0 - alpha, heatmap_color, alpha, 0)
                annotated[top_crop:bottom_crop, :] = eval_overlay

                # Extract Bounding Boxes around high-attribution clusters
                _, thresh_mask = cv2.threshold(heatmap_raw, int(255 * 0.65), 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                min_box_area = 250
                max_box_area = int(eval_w * eval_h * 0.45)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if min_box_area < area < max_box_area:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        # Map back to uncropped coordinates
                        full_by = by + top_crop
                        cv2.rectangle(annotated, (bx, full_by), (bx + bw, full_by + bh), (0, 0, 255), 2)
                        cv2.putText(annotated, "SOTIF OOD HAZARD", (bx, max(25, full_by - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # 7. Render Telemetry HUD Bar
        if is_hazard:
            status_text = "STATUS: CRITICAL SOTIF HAZARD (OOD)"
            status_color = (0, 0, 255)  # Red
        else:
            status_text = "STATUS: NOMINAL (IN-DISTRIBUTION)"
            status_color = (0, 255, 0)  # Green

        cv2.rectangle(annotated, (0, 0), (orig_w, 36), (20, 20, 20), -1)
        hud_line = f"{status_text} | MSE: {filt_mse:.5f} (th: {self.mse_th:.5f}) | D_M: {filt_dm:.2f} (th: {self.dm_th:.2f})"
        cv2.putText(annotated, hud_line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

        return annotated, raw_mse, filt_mse, raw_dm, filt_dm, is_hazard


def main():
    parser = argparse.ArgumentParser(description="Stage 06: Visual SOTIF Evaluation & Inspection")
    parser.add_argument("--dataset-key", type=str, default="a2d2", help="Key in configs/head_a.yaml for ROI crop settings")
    parser.add_argument("--source", type=str, required=True, help="Path to video file (.mp4/.avi) OR directory of frames")
    parser.add_argument("--output-dir", type=str, default="D:/SOTIF_supervision_workspace/evaluation_results")
    parser.add_argument("--show", action="store_true", help="Display interactive window during playback")
    args = parser.parse_args()

    evaluator = VisualSOTIFEvaluator()
    crop_cfg = evaluator.cfg.datasets.get(args.dataset_key, {}).get("roi_crop", {"enabled": False})

    source_path = Path(args.source)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Detect whether the input is a single video or image folder
    is_video_file = source_path.is_file() and source_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]

    if is_video_file:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {source_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INPUT] Video: {source_path.name} ({total_frames} frames @ {fps:.1f} FPS, {width}x{height})")
    elif source_path.is_dir():
        image_files = sorted(
            list(source_path.glob("*.webp")) + list(source_path.glob("*.png")) + list(source_path.glob("*.jpg")),
            key=natural_sort_key
        )
        if not image_files:
            raise FileNotFoundError(f"No image files found in {source_path}")
        sample = cv2.imread(str(image_files[0]))
        height, width, _ = sample.shape
        fps = 10.0
        total_frames = len(image_files)
        print(f"[INPUT] Directory: {len(image_files)} frames detected.")
    else:
        raise ValueError(f"Invalid source path: {source_path}")

    # 2. Output Video Writer
    video_out_path = out_dir / f"evaluated_{source_path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))

    records = []
    frame_idx = 0

    print(f"[INFO] Streaming inference and writing annotated video to: {video_out_path}")

    try:
        while True:
            # Read frame
            if is_video_file:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_name = f"frame_{frame_idx:06d}"
            else:
                if frame_idx >= len(image_files):
                    break
                img_p = image_files[frame_idx]
                frame = cv2.imread(str(img_p))
                frame_name = img_p.name

            # Forward pass through supervisor (backbone + Head A + Mahalanobis + Grad-CAM Heatmap)
            annotated, raw_mse, filt_mse, raw_dm, filt_dm, is_hazard = evaluator.inspect_frame(frame, crop_cfg)

            writer.write(annotated)

            records.append({
                "frame_idx": frame_idx,
                "frame_name": frame_name,
                "raw_mse": raw_mse,
                "filtered_mse": filt_mse,
                "raw_dm": raw_dm,
                "filtered_dm": filt_dm,
                "is_hazard": int(is_hazard),
            })

            # Optional GUI preview with safe headless fallback
            if args.show:
                try:
                    cv2.imshow("SOTIF Safety Supervisor Inspection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] Playback aborted by user.")
                        break
                except cv2.error:
                    print("[WARN] GUI display not supported by current environment. Disabling preview...")
                    args.show = False

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"[PROGRESS] Processed {frame_idx}/{total_frames} frames...")

    finally:
        if is_video_file:
            cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # 3. Export Telemetry CSV and Summary JSON
    df = pd.DataFrame(records)
    csv_path = out_dir / f"telemetry_{source_path.stem}.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "source": str(source_path),
        "total_frames_evaluated": len(df),
        "total_hazards_flagged": int(df["is_hazard"].sum()),
        "hazard_flag_percentage": float(df["is_hazard"].mean() * 100),
        "mean_filtered_mse": float(df["filtered_mse"].mean()),
        "mean_filtered_dm": float(df["filtered_dm"].mean()),
        "mse_threshold": evaluator.mse_th,
        "dm_threshold": evaluator.dm_th,
    }

    with open(out_dir / f"summary_{source_path.stem}.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 70)
    print("VIDEO EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Annotated Video:       {video_out_path}")
    print(f"Telemetry CSV:         {csv_path}")
    print(f"Summary JSON:          {out_dir / f'summary_{source_path.stem}.json'}")
    print(f"Hazards Flagged:       {summary['total_hazards_flagged']} / {summary['total_frames_evaluated']} frames ({summary['hazard_flag_percentage']:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()