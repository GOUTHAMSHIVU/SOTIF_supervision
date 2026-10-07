from pathlib import Path
from omegaconf import OmegaConf


def load_project_config(
    base_path: str = "configs/base.yaml", head_path: str = "configs/head_a.yaml"
) -> OmegaConf:
  base_cfg = (
      OmegaConf.load(base_path) if Path(base_path).exists() else OmegaConf.create()
  )
  head_cfg = (
      OmegaConf.load(head_path) if Path(head_path).exists() else OmegaConf.create()
  )
  cfg = OmegaConf.merge(base_cfg, head_cfg)
  OmegaConf.resolve(cfg)
  return cfg