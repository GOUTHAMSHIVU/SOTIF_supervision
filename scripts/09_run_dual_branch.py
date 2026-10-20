"""Run the existing VAE branch and complete YOLO branch on the same video/frames.

Run from the project root. Outputs are written to one folder:
  - dual_branch_<source>.xlsx (Frame_Summary and Object_Detections sheets)
  - frame_summary_<source>.csv
  - object_detections_<source>.csv
  - dual_branch_<source>.jsonl
  - evaluated_<source>.mp4 (if --save-video)
  - run_metadata_<source>.json

This is a synchronized inference/telemetry script, not a trained SOTIF supervisor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.head_a_vae import BetaVAE
from src.models.perception_yolo import YOLOPerception
from utils.config import load_project_config

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_sort_key(value: str):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VAE and complete YOLO branches on the same source and export synchronized telemetry."
    )
    parser.add_argument("--source", required=True, help="Video file or folder of images")
    parser.add_argument("--output-dir", default="outputs/dual_branch", help="Where to save outputs")
    parser.add_argument("--config", default="configs/head_a.yaml", help="Head-A YAML config")
    parser.add_argument("--dataset-key", default="a2d2", help="Dataset key used for configured ROI crop; use a key with crop disabled for external video")
    parser.add_argument("--yolo-weights", default="yolo11n.pt", help="Full YOLO checkpoint")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--device", default="auto", help="YOLO device: auto, cpu, 0, 1, etc.")
    parser.add_argument("--save-video", action="store_true", help="Write annotated output video")
    parser.add_argument("--show", action="store_true", help="Show an interactive preview")
    parser.add_argument(
        "--scenario-label",
        default="unspecified",
        choices=["unspecified", "id_normal", "domain_shift_normal", "semantic_anomaly", "domain_shift_semantic_anomaly"],
        help="Metadata label for later analysis; this does not train a supervisor.",
    )
    parser.add_argument("--scenario-id", default="", help="Optional manually assigned scenario/recording ID")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit; 0 processes all frames")
    return parser.parse_args()


def to_plain_dict(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "items"):
        return {str(k): v for k, v in value.items()}
    return {}


def crop_frame(frame_bgr: np.ndarray, crop_cfg: Any):
    """Match the vertical ROI-crop semantics in 06_evaluate_and_visualize.py."""
    orig_h, orig_w = frame_bgr.shape[:2]
    enabled = bool(crop_cfg.get("enabled", False)) if hasattr(crop_cfg, "get") else False
    if not enabled:
        return frame_bgr.copy(), 0, orig_h

    top_ratio = float(crop_cfg.get("top_ratio", 0.0))
    bottom_ratio = float(crop_cfg.get("bottom_ratio", 1.0))
    top = max(0, min(orig_h - 1, int(orig_h * top_ratio)))
    bottom = max(top + 1, min(orig_h, int(orig_h * bottom_ratio)))
    return frame_bgr[top:bottom, :].copy(), top, bottom


def get_frame_position(cx_norm: float, cy_norm: float) -> str:
    horizontal = "left" if cx_norm < (1 / 3) else "center" if cx_norm < (2 / 3) else "right"
    vertical = "top" if cy_norm < (1 / 3) else "middle" if cy_norm < (2 / 3) else "bottom"
    return f"{vertical}-{horizontal}"


def resolve_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class DualBranchRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = resolve_path(args.config)
        base_path = PROJECT_ROOT / "configs" / "base.yaml"
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Head config not found: {config_path}")

        self.cfg = load_project_config(
            base_path=str(base_path),
            head_path=str(config_path),
        )
        self.crop_cfg = self.cfg.datasets.get(args.dataset_key, {}).get(
            "roi_crop", {"enabled": False}
        )

        # ------------------------- VAE branch -------------------------
        # Intentionally preserves the current VAE branch's ResNet-50 feature
        # extractor, 256x512 transform, checkpoint, and calibration artifacts.
        resnet_weights = tv_models.ResNet50_Weights.DEFAULT
        resnet = tv_models.resnet50(weights=resnet_weights)
        self.conv_backbone = torch.nn.Sequential(*list(resnet.children())[:-2]).to(self.device).eval()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1)).to(self.device).eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 512), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        target_beta = self.cfg.evaluation.get("target_beta", 0.0001)
        models_dir = resolve_path(self.cfg.storage.models_dir)
        checkpoint_path = models_dir / f"head_a_beta{target_beta}_best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = models_dir / "head_a_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"VAE checkpoint not found. Tried configured path and fallback: {checkpoint_path}. "
                "Check configs/head_a.yaml and models_dir."
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.vae = BetaVAE(
            in_features=self.cfg.model.input_dim,
            latent_dim=self.cfg.model.latent_dim,
        ).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.vae.load_state_dict(state_dict)
        self.vae.eval()

        params_path = resolve_path(self.cfg.evaluation.calibrated_params)
        if not params_path.exists():
            raise FileNotFoundError(
                f"Calibrated Mahalanobis parameters not found: {params_path}. Run Stage 03 first."
            )
        with np.load(params_path) as params:
            self.mu_nom = np.asarray(params["mu"], dtype=np.float64)
            self.precision = np.asarray(params["precision"], dtype=np.float64)
            self.mse_threshold = float(params["mse_threshold_95"]) if "mse_threshold_95" in params else float(params.get("mse_threshold", 0.007843))
            self.dm_threshold = float(params["dm_threshold_95"]) if "dm_threshold_95" in params else float(params.get("dm_threshold", 5.9761))

        self.window_size = int(self.cfg.evaluation.temporal.get("window_size", 5))
        self.mse_buffer: list[float] = []
        self.dm_buffer: list[float] = []
        self.checkpoint_path = checkpoint_path
        self.params_path = params_path

        # ------------------------- YOLO branch ------------------------
        yolo_device = args.device
        if yolo_device == "auto":
            yolo_device = "0" if torch.cuda.is_available() else "cpu"
        elif yolo_device.lower() == "cuda":
            yolo_device = "0" if torch.cuda.is_available() else "cpu"

        self.yolo = YOLOPerception(
            weights=args.yolo_weights,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=yolo_device,
        )
        self.yolo_device = yolo_device

        print(f"[VAE] device={self.device}; checkpoint={self.checkpoint_path}")
        print(f"[VAE] calibration={self.params_path}")
        print(f"[VAE] thresholds: MSE={self.mse_threshold:.6f}, DM={self.dm_threshold:.4f}")
        print(f"[YOLO] weights={args.yolo_weights}; device={self.yolo_device}")
        print(f"[CROP] dataset_key={args.dataset_key}; crop={to_plain_dict(self.crop_cfg)}")

    def reset_temporal_buffers(self):
        self.mse_buffer.clear()
        self.dm_buffer.clear()

    @torch.inference_mode()
    def score_vae(self, eval_region: np.ndarray) -> dict:
        rgb = cv2.cvtColor(eval_region, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        feat_map = self.conv_backbone(tensor)
        x_gap = self.global_pool(feat_map).squeeze(-1).squeeze(-1)
        recon, mu, logvar, _z = self.vae(x_gap)

        raw_mse = float(F.mse_loss(recon, x_gap, reduction="mean").item())
        raw_kl = float((-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item())
        mu_np = mu.squeeze(0).detach().cpu().numpy().astype(np.float64)
        delta = mu_np - self.mu_nom
        dm_sq = float(delta @ self.precision @ delta)
        raw_dm = float(np.sqrt(max(dm_sq, 0.0)))

        self.mse_buffer.append(raw_mse)
        self.dm_buffer.append(raw_dm)
        if len(self.mse_buffer) > self.window_size:
            self.mse_buffer.pop(0)
            self.dm_buffer.pop(0)

        filtered_mse = float(np.mean(self.mse_buffer))
        filtered_dm = float(np.mean(self.dm_buffer))
        # Matches the current evaluator's AND gate. This is a baseline OOD
        # indicator, not a validated SOTIF hazard decision.
        baseline_ood = int(
            (filtered_mse > self.mse_threshold) and
            (filtered_dm > self.dm_threshold)
        )

        return {
            "raw_mse": raw_mse,
            "filtered_mse": filtered_mse,
            "raw_kl": raw_kl,
            "raw_dm": raw_dm,
            "filtered_dm": filtered_dm,
            "mse_threshold": self.mse_threshold,
            "dm_threshold": self.dm_threshold,
            "mse_exceeds_threshold": int(filtered_mse > self.mse_threshold),
            "dm_exceeds_threshold": int(filtered_dm > self.dm_threshold),
            "baseline_ood_flag": baseline_ood,
            "latent_mu_mean": float(np.mean(mu_np)),
            "latent_mu_std": float(np.std(mu_np)),
            "latent_sigma_mean": float(torch.exp(0.5 * logvar).mean().item()),
            "latent_sigma_std": float(torch.exp(0.5 * logvar).std(unbiased=False).item()),
        }

    def run(self):
        source = Path(self.args.source).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Input source does not exist: {source}")

        out_dir = Path(self.args.output_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        is_video = source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS
        if source.is_dir():
            image_paths = sorted(
                [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
                key=natural_sort_key,
            )
            if not image_paths:
                raise FileNotFoundError(f"No supported images found in directory: {source}")
            cap = None
            sample = cv2.imread(str(image_paths[0]))
            if sample is None:
                raise RuntimeError(f"Could not read image: {image_paths[0]}")
            height, width = sample.shape[:2]
            fps = 10.0
            total_frames = len(image_paths)
        elif is_video:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {source}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            image_paths = []
        else:
            raise ValueError(f"Source must be a supported video or an image directory: {source}")

        stem = source.stem if source.is_file() else source.name
        frame_csv_path = out_dir / f"frame_summary_{stem}.csv"
        object_csv_path = out_dir / f"object_detections_{stem}.csv"
        xlsx_path = out_dir / f"dual_branch_{stem}.xlsx"
        jsonl_path = out_dir / f"dual_branch_{stem}.jsonl"
        metadata_path = out_dir / f"run_metadata_{stem}.json"
        video_path = out_dir / f"evaluated_{stem}.mp4"

        video_writer = None
        if self.args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            if not video_writer.isOpened():
                raise RuntimeError(f"Could not create output video: {video_path}")

        frame_rows = []
        object_rows = []
        frame_idx = 0
        self.reset_temporal_buffers()

        try:
            with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
                while True:
                    if self.args.max_frames and frame_idx >= self.args.max_frames:
                        break

                    if is_video:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            break
                        timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                        frame_name = f"frame_{frame_idx:06d}"
                    else:
                        if frame_idx >= len(image_paths):
                            break
                        image_path = image_paths[frame_idx]
                        frame_bgr = cv2.imread(str(image_path))
                        if frame_bgr is None:
                            print(f"[WARN] Skipping unreadable image: {image_path}")
                            frame_idx += 1
                            continue
                        timestamp_ms = float(frame_idx * 1000.0 / fps)
                        frame_name = image_path.name

                    orig_h, orig_w = frame_bgr.shape[:2]
                    eval_region, top_crop, bottom_crop = crop_frame(frame_bgr, self.crop_cfg)
                    eval_h, eval_w = eval_region.shape[:2]

                    # Both branches receive the same cropped scene.
                    vae_result = self.score_vae(eval_region)
                    detections = self.yolo.predict(eval_region)

                    confidences = [float(d.confidence) for d in detections]
                    object_count = len(detections)
                    mean_conf = float(np.mean(confidences)) if confidences else 0.0
                    max_conf = float(np.max(confidences)) if confidences else 0.0
                    min_conf = float(np.min(confidences)) if confidences else 0.0
                    low_conf_count = int(sum(c < 0.5 for c in confidences))
                    class_counts: dict[str, int] = {}
                    for det in detections:
                        class_counts[str(det.class_name)] = class_counts.get(str(det.class_name), 0) + 1

                    frame_row = {
                        "source": str(source),
                        "scenario_id": self.args.scenario_id or source.stem,
                        "scenario_label": self.args.scenario_label,
                        "frame_idx": frame_idx,
                        "frame_name": frame_name,
                        "timestamp_ms": timestamp_ms,
                        "original_width": orig_w,
                        "original_height": orig_h,
                        "eval_width": eval_w,
                        "eval_height": eval_h,
                        "crop_top_px": top_crop,
                        "crop_bottom_px": bottom_crop,
                        **vae_result,
                        "yolo_object_count": object_count,
                        "yolo_mean_confidence": mean_conf,
                        "yolo_max_confidence": max_conf,
                        "yolo_min_confidence": min_conf,
                        "yolo_low_confidence_count_lt_0_5": low_conf_count,
                        "yolo_class_counts_json": json.dumps(class_counts, sort_keys=True),
                    }
                    frame_rows.append(frame_row)

                    frame_detection_records = []
                    for object_idx, det in enumerate(detections):
                        # Ultralytics coordinates refer to eval_region. Convert y coordinates
                        # back to the original, uncropped image coordinate system.
                        x1 = float(np.clip(det.x1, 0, eval_w - 1))
                        y1_crop = float(np.clip(det.y1, 0, eval_h - 1))
                        x2 = float(np.clip(det.x2, 0, eval_w - 1))
                        y2_crop = float(np.clip(det.y2, 0, eval_h - 1))
                        y1 = y1_crop + top_crop
                        y2 = y2_crop + top_crop
                        bw = max(0.0, x2 - x1)
                        bh = max(0.0, y2 - y1)
                        area = bw * bh
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        cx_norm = cx / max(orig_w, 1)
                        cy_norm = cy / max(orig_h, 1)
                        obj_row = {
                            "source": str(source),
                            "scenario_id": self.args.scenario_id or source.stem,
                            "scenario_label": self.args.scenario_label,
                            "frame_idx": frame_idx,
                            "timestamp_ms": timestamp_ms,
                            "object_idx": object_idx,
                            "class_id": int(det.class_id),
                            "class_name": str(det.class_name),
                            "confidence": float(det.confidence),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "width": bw, "height": bh, "area": area,
                            "center_x": cx, "center_y": cy,
                            "x1_norm": x1 / max(orig_w, 1),
                            "y1_norm": y1 / max(orig_h, 1),
                            "x2_norm": x2 / max(orig_w, 1),
                            "y2_norm": y2 / max(orig_h, 1),
                            "width_norm": bw / max(orig_w, 1),
                            "height_norm": bh / max(orig_h, 1),
                            "area_norm": area / max(orig_w * orig_h, 1),
                            "center_x_norm": cx_norm,
                            "center_y_norm": cy_norm,
                            "frame_position": get_frame_position(cx_norm, cy_norm),
                        }
                        object_rows.append(obj_row)
                        frame_detection_records.append(obj_row)

                    jsonl_file.write(json.dumps({
                        "source": str(source),
                        "scenario_id": self.args.scenario_id or source.stem,
                        "scenario_label": self.args.scenario_label,
                        "frame_idx": frame_idx,
                        "frame_name": frame_name,
                        "timestamp_ms": timestamp_ms,
                        "image_width": orig_w,
                        "image_height": orig_h,
                        "vae": vae_result,
                        "yolo_object_count": object_count,
                        "detections": frame_detection_records,
                    }, ensure_ascii=False) + "\n")

                    # Annotate detections and show the baseline OOD state, never label it as a validated SOTIF hazard.
                    if video_writer is not None or self.args.show:
                        annotated = frame_bgr.copy()
                        for det, obj in zip(detections, frame_detection_records):
                            x1, y1, x2, y2 = (int(round(obj[k])) for k in ("x1", "y1", "x2", "y2"))
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{det.class_name} {det.confidence:.2f}"
                            cv2.putText(annotated, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        status = "BASELINE OOD" if vae_result["baseline_ood_flag"] else "BASELINE OOD GATE: NOT TRIGGERED"
                        color = (0, 0, 255) if vae_result["baseline_ood_flag"] else (0, 220, 0)
                        hud = (
                            f"{status} | MSE {vae_result['filtered_mse']:.5f}/"
                            f"{self.mse_threshold:.5f} | DM {vae_result['filtered_dm']:.2f}/"
                            f"{self.dm_threshold:.2f} | YOLO objs {object_count}"
                        )
                        cv2.rectangle(annotated, (0, 0), (orig_w, 34), (20, 20, 20), -1)
                        cv2.putText(annotated, hud[:180], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                        if video_writer is not None:
                            video_writer.write(annotated)
                        if self.args.show:
                            try:
                                cv2.imshow("VAE + YOLO dual branch (baseline only)", annotated)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    print("[INFO] Stopped by user.")
                                    break
                            except cv2.error:
                                print("[WARN] OpenCV preview unavailable; continuing headless.")
                                self.args.show = False

                    frame_idx += 1
                    if frame_idx % 50 == 0:
                        pct = (frame_idx / total_frames * 100.0) if total_frames else 0.0
                        print(f"[PROGRESS] {frame_idx}/{total_frames or '?'} frames ({pct:.1f}%)")
        finally:
            if cap is not None:
                cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()

        frame_df = pd.DataFrame(frame_rows)
        object_columns = [
            "source", "scenario_id", "scenario_label", "frame_idx", "timestamp_ms", "object_idx",
            "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "width", "height", "area",
            "center_x", "center_y", "x1_norm", "y1_norm", "x2_norm", "y2_norm", "width_norm", "height_norm",
            "area_norm", "center_x_norm", "center_y_norm", "frame_position",
        ]
        object_df = pd.DataFrame(object_rows, columns=object_columns)
        frame_df.to_csv(frame_csv_path, index=False)
        object_df.to_csv(object_csv_path, index=False)

        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                frame_df.to_excel(writer, sheet_name="Frame_Summary", index=False)
                object_df.to_excel(writer, sheet_name="Object_Detections", index=False)
        except ImportError:
            print("[WARN] openpyxl not installed; CSV outputs are available. Install with: pip install openpyxl")

        metadata = {
            "source": str(source),
            "scenario_id": self.args.scenario_id or source.stem,
            "scenario_label": self.args.scenario_label,
            "frames_processed": int(len(frame_df)),
            "object_detections": int(len(object_df)),
            "vae_checkpoint": str(self.checkpoint_path),
            "calibration_file": str(self.params_path),
            "vae_device": str(self.device),
            "yolo_weights": self.args.yolo_weights,
            "yolo_device": self.yolo_device,
            "yolo_conf_threshold": self.args.conf,
            "yolo_iou_threshold": self.args.iou,
            "yolo_imgsz": self.args.imgsz,
            "mse_threshold": self.mse_threshold,
            "dm_threshold": self.dm_threshold,
            "baseline_ood_gate": "(filtered_mse > mse_threshold) AND (filtered_dm > dm_threshold)",
            "warning": "baseline_ood_flag is not a validated SOTIF hazard label; no trained supervisor is used in this script.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("\n" + "=" * 72)
        print("DUAL-BRANCH INFERENCE COMPLETE")
        print("=" * 72)
        print(f"Frames processed : {len(frame_df)}")
        print(f"Object detections: {len(object_df)}")
        if len(frame_df):
            n_ood = int(frame_df["baseline_ood_flag"].sum())
            print(f"Baseline OOD flags: {n_ood}/{len(frame_df)} ({100*n_ood/len(frame_df):.2f}%)")
        print(f"Frame CSV        : {frame_csv_path}")
        print(f"Object CSV       : {object_csv_path}")
        print(f"Excel workbook   : {xlsx_path} (if openpyxl is installed)")
        print(f"JSONL            : {jsonl_path}")
        print(f"Run metadata     : {metadata_path}")
        if self.args.save_video:
            print(f"Annotated video  : {video_path}")
        print("Note: the OOD gate is a baseline score, not a validated SOTIF decision.")
        print("=" * 72)


def main():
    args = parse_args()
    runner = DualBranchRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
