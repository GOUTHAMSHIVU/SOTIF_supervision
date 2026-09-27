# src/utils/manifest.py
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import h5py
import torch

def compute_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """Calculates SHA-256 cryptographic hash of the generated file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "UNKNOWN_COMMIT"

def write_artifact_manifest(h5_path: str, raw_data_dir: str, output_manifest_path: str):
    h5_p = Path(h5_path)
    file_bytes = h5_p.stat().st_size
    file_hash = compute_sha256(h5_path)
    
    with h5py.File(h5_path, "r") as f:
        num_samples = f["features"].shape[0]
        feature_dim = f["features"].shape[1]
        dtype = str(f["features"].dtype)

    manifest_record = {
        "artifact_name": h5_p.name,
        "artifact_path": str(h5_p.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit(),
        "torch_version": torch.__version__,
        "dataset_provenance": {
            "source_dir": str(Path(raw_data_dir).resolve()),
            "total_samples": num_samples,
            "feature_dim": feature_dim
        },
        "tensor_spec": {
            "shape": [num_samples, feature_dim],
            "dtype": dtype,
            "size_mb": round(file_bytes / (1024 * 1024), 2)
        },
        "integrity": {
            "algorithm": "SHA-256",
            "sha256": file_hash
        }
    }

    # 1. Write the static snapshot manifest
    out_p = Path(output_manifest_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest_record, f, indent=2)

    # 2. Append to continuous audit ledger (.jsonl)
    audit_ledger_p = out_p.parent / "data_provenance.jsonl"
    with open(audit_ledger_p, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_record) + "\n")

    print(f"[AUDIT] Manifest generated: {out_p}")
    print(f"[AUDIT] SHA-256: {file_hash}")