import os
import sys
import subprocess
from pathlib import Path
import yaml
from omegaconf import OmegaConf

def setup_a2d2_link(config_path: str = "configs/base.yaml"):
    # 1. Load config using OmegaConf to resolve ${storage.data_dir} variables
    cfg = OmegaConf.load(config_path)
    
    # Resolve target raw path from YAML
    # e.g., "${storage.data_dir}/raw" -> "D:/sotif_workspace/data/raw"
    target_raw_root = Path(cfg.storage.data_dir) / "raw"
    target_a2d2 = target_raw_root / "a2d2"

    if not target_a2d2.exists():
        # Fallback check if images are directly inside target_raw_root
        if target_raw_root.exists() and list(target_raw_root.rglob("*.png")):
            target_a2d2 = target_raw_root
        else:
            raise FileNotFoundError(
                f"[ERROR] Could not find A2D2 images under: {target_a2d2.resolve()} or {target_raw_root.resolve()}.\n"
                f"Please verify 'storage.data_dir' in {config_path}."
            )

    # 2. Local link destination inside the repo
    repo_root = Path(__file__).resolve().parent.parent
    local_raw_dir = repo_root / "data" / "raw"
    local_link = local_raw_dir / "a2d2"

    local_raw_dir.mkdir(parents=True, exist_ok=True)

    if local_link.exists() or local_link.is_symlink():
        print(f"[INFO] Link already exists at: {local_link}")
        return local_link

    target_str = str(target_a2d2.resolve())
    link_str = str(local_link.resolve())

    print(f"[INFO] Creating junction link:")
    print(f"       From: {link_str}")
    print(f"       To:   {target_str}")

    # 3. Create Windows Directory Junction
    if sys.platform == "win32":
        cmd = f'mklink /J "{link_str}" "{target_str}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to create junction: {res.stderr}")
    else:
        os.symlink(target_str, link_str)

    print(f"[SUCCESS] Junction successfully linked: {local_link} -> {target_a2d2}")
    return local_link

if __name__ == "__main__":
    setup_a2d2_link()