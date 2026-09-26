"""
SOTIF Perception Supervisor - Workspace & Configuration Sanity Check
Validates:
  1. python-dotenv loading of STORAGE_ROOT
  2. OmegaConf dynamic variable interpolation
  3. Physical read/write permissions on the D: drive workspace
  4. CUDA availability on the active PyTorch installation
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf
import torch


def run_checks(config_path: str = "configs/base.yaml"):
    print("=" * 70)
    print("   SOTIF_supervision: Environment & Path Resolution Audit   ")
    print("=" * 70)

    # 1. Load local .env
    env_loaded = load_dotenv()
    env_root = os.getenv("STORAGE_ROOT")
    print(f"[1/4] .env Loaded: {env_loaded} | STORAGE_ROOT = '{env_root}'")
    if not env_root:
        print("  --> WARNING: STORAGE_ROOT not in .env. Will rely on YAML default.")

    # 2. Check config file existence
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"  --> FATAL: Configuration file not found at {cfg_file.resolve()}")
        sys.exit(1)

    # 3. Load & Resolve OmegaConf
    cfg = OmegaConf.load(cfg_file)
    resolved_dict = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg = OmegaConf.create(resolved_dict)
    
    workspace_dir = Path(resolved_cfg.storage.root)
    models_dir = Path(resolved_cfg.storage.models_dir)
    data_proc_dir = Path(resolved_cfg.storage.processed_dir)
    mlflow_db = resolved_cfg.storage.mlflow.tracking_uri

    print("\n[2/4] Resolved Configuration Paths:")
    print(f"  • Workspace Root : {workspace_dir.resolve()}")
    print(f"  • Models Dir     : {models_dir.resolve()}")
    print(f"  • Processed Data : {data_proc_dir.resolve()}")
    print(f"  • MLflow URI     : {mlflow_db}")

    # 4. Verify Physical Disk Permissions on Target Drive
    print(f"\n[3/4] Testing Read/Write Access on Target Drive ({workspace_dir.drive or workspace_dir})...")
    try:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        data_proc_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = workspace_dir / ".io_canary_test.tmp"
        with open(test_file, "w") as f:
            f.write("SOTIF_supervision IO canary check passed.")
        
        with open(test_file, "r") as f:
            canary_text = f.read()

        test_file.unlink()  # Clean up temporary test file
        print("  --> Read/Write/Delete cycle: SUCCESS (Target disk operational)")
    except Exception as e:
        print(f"  --> FATAL: Disk permission or path error: {e}")
        sys.exit(1)

    # 5. Device & PyTorch CUDA Sanity
    print("\n[4/4] Hardware & Runtime Check:")
    cuda_available = torch.cuda.is_available()
    print(f"  • PyTorch Version : {torch.__version__}")
    print(f"  • CUDA Available  : {cuda_available}")
    if cuda_available:
        print(f"  • Active Device   : {torch.cuda.get_device_name(0)}")
        print(f"  • VRAM Allocated  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("  • Notice: Running in CPU mode (local). Cloud execution will use L4 GPU.")

    print("\n" + "=" * 70)
    print("   ALL PRE-FLIGHT CHECKS PASSED. Ready for feature extraction.   ")
    print("=" * 70)


if __name__ == "__main__":
    run_checks()