import cv2
import numpy as np


def draw_hud(
    frame: np.ndarray,
    mse: float,
    mse_th: float,
    dm: float,
    dm_th: float,
    is_hazard: bool,
) -> np.ndarray:
  """Draws semi-transparent SOTIF telemetry HUD on frame."""
  h, w, _ = frame.shape
  overlay = frame.copy()
  cv2.rectangle(overlay, (0, 0), (w, 80), (15, 15, 15), -1)
  cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

  status_text = (
      "STATUS: CRITICAL SOTIF HAZARD"
      if is_hazard
      else "STATUS: NOMINAL (IN-DISTRIBUTION)"
  )
  color = (0, 0, 255) if is_hazard else (0, 255, 0)
  cv2.putText(
      frame, status_text, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2
  )

  metrics = (
      f"MSE: {mse:.5f} (th: {mse_th:.5f})  |  D_M: {dm:.2f} (th: {dm_th:.2f})"
  )
  cv2.putText(
      frame,
      metrics,
      (20, 62),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.52,
      (230, 230, 230),
      1,
  )
  return frame