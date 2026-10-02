# utils/manifest.py
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
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

def write_artifact_manifest(
    h5_path: str,
    raw_data_dir: str,
    output_manifest_path: str = "data/metadata/nominal_train_manifest_post.json"
):
    import h5py
    h5_p = Path(h5_path)
    if not h5_p.exists():
        raise FileNotFoundError(f"[ERROR] Target .h5 file not found: {h5_p.resolve()}")

    file_bytes = h5_p.stat().st_size
    file_hash = compute_sha256(str(h5_p))
    
    with h5py.File(h5_p, "r") as f:
        num_samples = f["features"].shape[0]
        feature_dim = f["features"].shape[1]
        dtype = str(f["features"].dtype)

    manifest_record = {
        "status": "COMPLETED_EXTRACTION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "source_pre_manifest": "data/metadata/nominal_train_manifest_pre.json",
        "dataset_provenance": {
            "source_dir": str(Path(raw_data_dir).resolve()),
            "total_samples": num_samples,
            "feature_dim": feature_dim
        },
        "artifact": {
            "name": h5_p.name,
            "path": str(h5_p.resolve()),
            "shape": [num_samples, feature_dim],
            "dtype": dtype,
            "size_mb": round(file_bytes / (1024 * 1024), 2)
        },
        "integrity": {
            "algorithm": "SHA-256",
            "sha256": file_hash
        }
    }

    out_p = Path(output_manifest_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest_record, f, indent=2)

    # Append to append-only audit trail
    ledger_p = out_p.parent / "data_provenance.jsonl"
    with open(ledger_p, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_record) + "\n")

    print(f"[AUDIT] Post-extraction manifest created: {out_p.resolve()}")
    print(f"[AUDIT] SHA-256: {file_hash}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset & Feature Manifest Tool")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to raw A2D2 dataset")
    parser.add_argument("--h5-path", type=str, default=None, help="Path to generated .h5 file")
    parser.add_argument("--out", type=str, default="data/metadata/nominal_train_manifest_post.json")
    args = parser.parse_args()

    if args.h5_path:
        write_artifact_manifest(args.h5_path, args.data_dir, args.out)
    else:
        print("[WARN] Run with --h5-path after extraction completes to generate post-manifest.")