from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseTracker(ABC):
    """Abstract interface for experiment tracking."""

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None):
        pass

    @abstractmethod
    def end_run(self):
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()