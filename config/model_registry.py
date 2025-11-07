from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "model_ids.json"
ENV_CONFIG_PATH = "KOOP_MODEL_IDS_PATH"


@lru_cache()
def _load_config(path: Optional[Path] = None) -> Dict[str, str]:
    config_path = path or Path(os.environ.get(ENV_CONFIG_PATH, DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise FileNotFoundError(f"Model config niet gevonden: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_model_id(key: str, *, config_path: Optional[Path] = None) -> str:
    config = _load_config(config_path)
    model_id = config.get(key)
    if not model_id:
        raise KeyError(f"Model ID '{key}' niet gevonden in {config_path or DEFAULT_CONFIG_PATH}")
    return model_id
