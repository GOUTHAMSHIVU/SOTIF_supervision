from dataclasses import dataclass, asdict
from typing import List

import numpy as np

from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def to_dict(self):
        return asdict(self) | {
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "center_x": self.center_x,
            "center_y": self.center_y,
        }


class YOLOPerception:

    def __init__(
        self,
        weights: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cuda",
    ):
        self.model = YOLO(weights)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:

        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        names = result.names

        for box, confidence, class_id in zip(
            boxes,
            confidences,
            classes,
        ):
            x1, y1, x2, y2 = box.tolist()

            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=str(names[int(class_id)]),
                    confidence=float(confidence),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )

        return detections