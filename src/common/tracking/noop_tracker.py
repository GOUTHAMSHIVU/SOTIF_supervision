from typing import Any, Dict, Optional
from src.common.tracking.base import BaseTracker

class NoOpTracker(BaseTracker):
    """Null-object pattern tracker when telemetry is disabled or during CI/offline runs."""

    def start_run(self, run_name: Optional[str] = None):
        pass

    def end_run(self):
        pass

    def log_params(self, params: Dict[str, Any]):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass