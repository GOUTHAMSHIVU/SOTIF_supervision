from typing import Any, Dict, Optional
from pathlib import Path
from src.common.tracking.base import BaseTracker

class MLflowTracker(BaseTracker):
    """MLflow concrete implementation."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None, run_name: Optional[str] = None):
        import mlflow
        self.mlflow = mlflow
        self.experiment_name = experiment_name
        self.run_name = run_name

        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)
        self.active_run = None

    def start_run(self, run_name: Optional[str] = None):
        name = run_name or self.run_name
        self.active_run = self.mlflow.start_run(run_name=name)
        return self.active_run

    def end_run(self):
        if self.mlflow.active_run():
            self.mlflow.end_run()
        self.active_run = None

    def log_params(self, params: Dict[str, Any]):
        # Sanitize non-serializable objects
        clean_params = {k: str(v) if isinstance(v, (Path, list, tuple)) else v for k, v in params.items()}
        self.mlflow.log_params(clean_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        p = Path(local_path)
        if p.exists():
            self.mlflow.log_artifact(str(p), artifact_path=artifact_path)