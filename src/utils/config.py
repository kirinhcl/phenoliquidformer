"""Configuration system using OmegaConf.

Usage:
    from src.utils.config import load_config
    cfg = load_config('configs/default.yaml')
    cfg = load_config('configs/default.yaml', overrides=['training.lr=3e-4'])
"""

from pathlib import Path
from typing import Any, Optional, Sequence

from omegaconf import DictConfig, OmegaConf


def load_config(
    path: str,
    overrides: Optional[Sequence[str]] = None,
    base_path: Optional[str] = None,
) -> DictConfig:
    """Load YAML config with optional base-config merging and CLI dot-list overrides.

    Base config is merged *under* the loaded config (loaded takes priority).
    """
    cfg = OmegaConf.load(path)

    if base_path is not None:
        base_cfg = OmegaConf.load(base_path)
        cfg = OmegaConf.merge(base_cfg, cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_cfg)

    assert isinstance(cfg, DictConfig)
    return cfg


def save_config(cfg: DictConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(container, dict)
    return {str(k): v for k, v in container.items()}
