import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.models.perception_yolo import YOLOPerception
from utils.frame_preprocess import apply_crop


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="YOLO perception extraction for SOTIF project"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "--weights",
        default="yolo11n.pt",
        help="YOLO checkpoint",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/perception",
        help="Output directory",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda / cpu / 0 / 1 etc.",
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated YOLO video",
    )

    return parser.parse_args()


# ============================================================
# FRAME LOCATION HELPER
# ============================================================

def get_frame_position(
    center_x_norm: float,
    center_y_norm: float,
) -> str:
    """
    Convert normalized object center coordinates into
    a human-readable 3x3 image position.
    """

    # Horizontal position
    if center_x_norm < 1 / 3:
        horizontal = "left"
    elif center_x_norm < 2 / 3:
        horizontal = "center"
    else:
        horizontal = "right"

    # Vertical position
    if center_y_norm < 1 / 3:
        vertical = "top"
    elif center_y_norm < 2 / 3:
        vertical = "middle"
    else:
        vertical = "bottom"

    return f"{vertical}-{horizontal}"


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    source = Path(args.source)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("SOTIF YOLO PERCEPTION EXTRACTION")
    print("=" * 70)
    print(f"Source       : {source}")
    print(f"YOLO weights : {args.weights}")
    print(f"Output dir   : {output_dir}")
    print(f"Device       : {args.device}")
    print(f"Confidence   : {args.conf}")
    print(f"IoU          : {args.iou}")
    print("=" * 70)

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    yolo = YOLOPerception(
        weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{source}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    original_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    original_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print()
    print(f"Video resolution : {original_width} x {original_height}")
    print(f"FPS              : {fps:.2f}")
    print(f"Total frames     : {total_frames}")
    print()

    # --------------------------------------------------------
    # Output filenames
    # --------------------------------------------------------

    xlsx_path = (
        output_dir /
        f"perception_{source.stem}.xlsx"
    )

    frame_csv_path = (
        output_dir /
        f"frame_summary_{source.stem}.csv"
    )

    object_csv_path = (
        output_dir /
        f"object_detections_{source.stem}.csv"
    )

    jsonl_path = (
        output_dir /
        f"detections_{source.stem}.jsonl"
    )

    video_writer = None

    # --------------------------------------------------------
    # Annotated video
    # --------------------------------------------------------

    if args.save_video:

        output_video = (
            output_dir /
            f"{source.stem}_yolo.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        video_writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            fps,
            (
                original_width,
                original_height,
            ),
        )

        if not video_writer.isOpened():
            raise RuntimeError(
                f"Could not create output video:\n"
                f"{output_video}"
            )

        print(
            f"Annotated video : {output_video}"
        )

    # --------------------------------------------------------
    # Data containers
    # --------------------------------------------------------

    frame_summary_rows = []
    object_detection_rows = []

    frame_idx = 0

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    try:

        while True:

            ret, frame_bgr = cap.read()

            if not ret:
                break

            # =================================================
            # SAME INPUT FRAME FOR VAE + YOLO
            # =================================================

            model_frame = apply_crop(
                frame_bgr,
                crop_cfg=None,
            )

            model_height, model_width = (
                model_frame.shape[:2]
            )

            # =================================================
            # YOLO
            # =================================================

            detections = yolo.predict(
                model_frame
            )

            # =================================================
            # FRAME-LEVEL STATISTICS
            # =================================================

            confidences = [
                float(d.confidence)
                for d in detections
            ]

            object_count = len(
                detections
            )

            if confidences:

                mean_confidence = float(
                    np.mean(confidences)
                )

                max_confidence = float(
                    np.max(confidences)
                )

                min_confidence = float(
                    np.min(confidences)
                )

                low_confidence_count = int(
                    sum(
                        c < 0.5
                        for c in confidences
                    )
                )

            else:

                mean_confidence = 0.0
                max_confidence = 0.0
                min_confidence = 0.0
                low_confidence_count = 0

            timestamp_ms = float(
                cap.get(
                    cv2.CAP_PROP_POS_MSEC
                )
            )

            # =================================================
            # FRAME SUMMARY ROW
            # =================================================

            frame_summary_rows.append({

                "frame_idx":
                    frame_idx,

                "timestamp_ms":
                    timestamp_ms,

                "object_count":
                    object_count,

                "mean_confidence":
                    mean_confidence,

                "max_confidence":
                    max_confidence,

                "min_confidence":
                    min_confidence,

                "low_confidence_count":
                    low_confidence_count,
            })

            # =================================================
            # OBJECT-LEVEL DATA
            # ONE ROW = ONE OBJECT
            # =================================================

            for object_idx, d in enumerate(
                detections
            ):

                # ---------------------------------------------
                # Bounding box
                # ---------------------------------------------

                x1 = float(d.x1)
                y1 = float(d.y1)
                x2 = float(d.x2)
                y2 = float(d.y2)

                # ---------------------------------------------
                # Clamp box to image dimensions
                # ---------------------------------------------

                x1 = max(
                    0.0,
                    min(x1, model_width - 1)
                )

                y1 = max(
                    0.0,
                    min(y1, model_height - 1)
                )

                x2 = max(
                    0.0,
                    min(x2, model_width - 1)
                )

                y2 = max(
                    0.0,
                    min(y2, model_height - 1)
                )

                # ---------------------------------------------
                # Geometry
                # ---------------------------------------------

                box_width = max(
                    0.0,
                    x2 - x1
                )

                box_height = max(
                    0.0,
                    y2 - y1
                )

                box_area = (
                    box_width *
                    box_height
                )

                # ---------------------------------------------
                # Center
                # ---------------------------------------------

                center_x = (
                    x1 + x2
                ) / 2.0

                center_y = (
                    y1 + y2
                ) / 2.0

                # ---------------------------------------------
                # Normalized coordinates
                # ---------------------------------------------

                x1_norm = (
                    x1 / model_width
                )

                y1_norm = (
                    y1 / model_height
                )

                x2_norm = (
                    x2 / model_width
                )

                y2_norm = (
                    y2 / model_height
                )

                width_norm = (
                    box_width /
                    model_width
                )

                height_norm = (
                    box_height /
                    model_height
                )

                area_norm = (
                    box_area /
                    (
                        model_width *
                        model_height
                    )
                )

                center_x_norm = (
                    center_x /
                    model_width
                )

                center_y_norm = (
                    center_y /
                    model_height
                )

                # ---------------------------------------------
                # Human-readable location
                # ---------------------------------------------

                frame_position = get_frame_position(
                    center_x_norm,
                    center_y_norm,
                )

                # ---------------------------------------------
                # Save object information
                # ---------------------------------------------

                object_detection_rows.append({

                    # Frame information
                    "frame_idx":
                        frame_idx,

                    "timestamp_ms":
                        timestamp_ms,

                    "object_idx":
                        object_idx,

                    # Object identity
                    "class_id":
                        int(d.class_id),

                    "class_name":
                        d.class_name,

                    "confidence":
                        float(d.confidence),

                    # Exact bounding box in pixels
                    "x1":
                        x1,

                    "y1":
                        y1,

                    "x2":
                        x2,

                    "y2":
                        y2,

                    # Bounding box dimensions
                    "width":
                        box_width,

                    "height":
                        box_height,

                    "area":
                        box_area,

                    # Object center in pixels
                    "center_x":
                        center_x,

                    "center_y":
                        center_y,

                    # Normalized bounding box
                    "x1_norm":
                        x1_norm,

                    "y1_norm":
                        y1_norm,

                    "x2_norm":
                        x2_norm,

                    "y2_norm":
                        y2_norm,

                    # Normalized dimensions
                    "width_norm":
                        width_norm,

                    "height_norm":
                        height_norm,

                    "area_norm":
                        area_norm,

                    # Normalized center
                    "center_x_norm":
                        center_x_norm,

                    "center_y_norm":
                        center_y_norm,

                    # Human-readable position
                    "frame_position":
                        frame_position,
                })

            # =================================================
            # JSONL
            # =================================================

            json_record = {

                "frame_idx":
                    frame_idx,

                "timestamp_ms":
                    timestamp_ms,

                "image_width":
                    model_width,

                "image_height":
                    model_height,

                "object_count":
                    object_count,

                "detections": [
                    d.to_dict()
                    for d in detections
                ],
            }

            # We open this later after processing
            # to avoid keeping a large file handle open.
            # Store temporary records separately.
            if frame_idx == 0:
                jsonl_records = []

            jsonl_records.append(
                json_record
            )

            # =================================================
            # ANNOTATED VIDEO
            # =================================================

            if video_writer is not None:

                annotated = frame_bgr.copy()

                for object_idx, d in enumerate(
                    detections
                ):

                    # Detection coordinates are relative
                    # to model_frame.
                    #
                    # Currently crop_cfg=None, so model_frame
                    # and frame_bgr have the same dimensions.

                    x1 = int(
                        max(0, d.x1)
                    )

                    y1 = int(
                        max(0, d.y1)
                    )

                    x2 = int(
                        min(
                            original_width - 1,
                            d.x2
                        )
                    )

                    y2 = int(
                        min(
                            original_height - 1,
                            d.y2
                        )
                    )

                    # -----------------------------------------
                    # Bounding box
                    # -----------------------------------------

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    # -----------------------------------------
                    # Label
                    # -----------------------------------------

                    label = (
                        f"{d.class_name} "
                        f"ID:{d.class_id} "
                        f"{d.confidence:.2f}"
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (
                            x1,
                            max(20, y1 - 8),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # -----------------------------------------
                    # Object center
                    # -----------------------------------------

                    center_x = int(
                        (d.x1 + d.x2) / 2
                    )

                    center_y = int(
                        (d.y1 + d.y2) / 2
                    )

                    cv2.circle(
                        annotated,
                        (
                            center_x,
                            center_y,
                        ),
                        4,
                        (0, 0, 255),
                        -1,
                    )

                # ---------------------------------------------
                # Frame information
                # ---------------------------------------------

                frame_text = (
                    f"Frame: {frame_idx} | "
                    f"Objects: {object_count}"
                )

                cv2.putText(
                    annotated,
                    frame_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                video_writer.write(
                    annotated
                )

            # =================================================
            # Progress
            # =================================================

            frame_idx += 1

            if frame_idx % 100 == 0:

                percentage = (
                    frame_idx /
                    total_frames *
                    100
                    if total_frames > 0
                    else 0
                )

                print(
                    f"Processed "
                    f"{frame_idx}/{total_frames} "
                    f"frames "
                    f"({percentage:.1f}%)"
                )

    finally:

        cap.release()

        if video_writer is not None:
            video_writer.release()

    # ========================================================
    # CREATE DATAFRAMES
    # ========================================================

    frame_df = pd.DataFrame(
        frame_summary_rows
    )

    object_df = pd.DataFrame(
        object_detection_rows
    )

    # ========================================================
    # CSV OUTPUT
    # ========================================================

    frame_df.to_csv(
        frame_csv_path,
        index=False,
    )

    object_df.to_csv(
        object_csv_path,
        index=False,
    )

    # ========================================================
    # EXCEL OUTPUT
    # ========================================================

    try:

        with pd.ExcelWriter(
            xlsx_path,
            engine="openpyxl",
        ) as writer:

            frame_df.to_excel(
                writer,
                sheet_name="Frame_Summary",
                index=False,
            )

            object_df.to_excel(
                writer,
                sheet_name="Object_Detections",
                index=False,
            )

    except ImportError:

        print()
        print(
            "openpyxl is not installed."
        )
        print(
            "Run:"
        )
        print(
            "pip install openpyxl"
        )

    # ========================================================
    # JSONL OUTPUT
    # ========================================================

    with open(
        jsonl_path,
        "w",
        encoding="utf-8",
    ) as f:

        for record in jsonl_records:

            f.write(
                json.dumps(record)
                + "\n"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_objects = len(
        object_detection_rows
    )

    detected_frames = int(
        (
            frame_df["object_count"] > 0
        ).sum()
    )

    no_detection_frames = int(
        (
            frame_df["object_count"] == 0
        ).sum()
    )

    print()
    print("=" * 70)
    print("YOLO PERCEPTION EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Frames processed      : {frame_idx}"
    )

    print(
        f"Frames with detection : {detected_frames}"
    )

    print(
        f"Frames without detect.: {no_detection_frames}"
    )

    print(
        f"Total objects detected: {total_objects}"
    )

    print()
    print("OUTPUT FILES")
    print("-" * 70)

    print(
        f"Excel          : {xlsx_path}"
    )

    print(
        f"Frame CSV      : {frame_csv_path}"
    )

    print(
        f"Object CSV     : {object_csv_path}"
    )

    print(
        f"JSONL          : {jsonl_path}"
    )

    if args.save_video:

        print(
            f"Annotated video: "
            f"{output_video}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()