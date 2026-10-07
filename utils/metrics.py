import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_mahalanobis(
    mu: np.ndarray, mu_nom: np.ndarray, precision: np.ndarray
) -> np.ndarray:
  """Computes Mahalanobis distance for a batch of latent vectors."""
  delta = mu - mu_nom
  return np.sqrt(np.sum(np.dot(delta, precision) * delta, axis=1))


def compute_sotif_metrics(
    y_true: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
  """Computes AUROC and FPR at 95% TPR for OOD benchmark evaluation."""
  auroc = float(roc_auc_score(y_true, scores))
  fpr, tpr, thresholds = roc_curve(y_true, scores)

  # FPR at 95% TPR
  idx = np.argmin(np.abs(tpr - 0.95))
  fpr95 = float(fpr[idx])

  return {"auroc": auroc, "fpr95": fpr95}