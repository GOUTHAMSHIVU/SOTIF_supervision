from pathlib import Path
import json
import numpy as np


def save_calibration_manifest(output_path: Path, data: dict):
  """Saves calibration statistics and SOTIF safety thresholds."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path.with_suffix(".npz"), **data)