from typing import Optional
from src.common.tracking.base import BaseTracker
from src.common.tracking.noop_tracker import NoOpTracker

def get_tracker(
    enabled: bool = False,
    experiment_name: str = "default_experiment",
    tracking_uri: Optional[str] = "http://localhost:5000",
    run_name: Optional[str] = None
) -> BaseTracker:
    """Factory helper to cleanly instantiate either MLflow or NoOp tracker."""
    if not enabled:
        return NoOpTracker()
    
    try:
        from src.common.tracking.mlflow_tracker import MLflowTracker
        return MLflowTracker(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_name=run_name
        )
    except Exception as e:
        print(f"[WARN] Failed to initialize MLflow tracker ({e}). Falling back to NoOpTracker.")
        return NoOpTracker()